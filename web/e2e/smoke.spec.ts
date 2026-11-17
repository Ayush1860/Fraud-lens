// The one end-to-end check: a flagged transaction reaches the queue, an analyst labels
// it in the browser, and the label is stored. Runs against the Compose stack.
import { expect, test, type APIRequestContext } from "@playwright/test";

const START = Date.UTC(2019, 5, 9, 2, 0, 0) / 1000; // a night-time burst, in Unix seconds
const AMOUNTS = [180, 420, 760, 1150, 1640, 2300, 4000, 9000];

/** Post a burst of escalating purchases on one fresh card; return the first flagged id. */
async function scoreBurst(request: APIRequestContext): Promise<string> {
  const run = Date.now().toString(16);
  const card = `c_${run.padStart(16, "0").slice(-16)}`;
  for (const [i, amount] of AMOUNTS.entries()) {
    const transactionId = `e2e-${run}-${i}`;
    const response = await request.post("/api/v1/transactions/score", {
      data: {
        transaction_id: transactionId,
        card_id: card,
        event_time: new Date((START + 300 * i) * 1000).toISOString(),
        amount,
        category: "shopping_net",
        merchant_id: "m_e2e",
        merchant_lat: 40.7,
        merchant_long: -74.0,
        home_lat: 40.7,
        home_long: -74.0,
        city_pop: 50000,
        cardholder_age: 35,
      },
    });
    expect(response.ok(), await response.text()).toBeTruthy();
    const body = (await response.json()) as { decision: string };
    if (body.decision !== "approve") return transactionId;
  }
  throw new Error("no transaction in the burst was flagged");
}

test("an alert is labelled in the browser and the label is stored", async ({ page, request }) => {
  const flagged = await scoreBurst(request);

  await page.goto("/");
  await expect(page.getByRole("note")).toContainText("Simulated data");
  const link = page.locator(`a[href^="/cases/${flagged}?alert="]`);
  await expect(link).toBeVisible({ timeout: 15_000 });
  const href = (await link.getAttribute("href")) ?? "";
  const alertId = Number(new URL(href, "http://x").searchParams.get("alert"));
  await link.click();

  await expect(page.getByRole("heading", { name: "Why it was flagged" })).toBeVisible();
  await expect(page.locator("ol.reasons li").first()).toBeVisible();
  await page.getByLabel("Analyst").fill("e2e");
  await page.getByLabel("Note").fill("smoke test");
  await page.getByRole("button", { name: "Confirm fraud" }).click();
  await expect(page.getByText("Recorded: confirmed fraud by e2e")).toBeVisible();

  // The label is in Postgres: the alert now reads back as closed from the service.
  const closed = await request.get("/api/v1/alerts?status=closed&limit=200");
  const items = ((await closed.json()) as { items: { id: number }[] }).items;
  expect(items.map((item) => item.id)).toContain(alertId);
});

test("the other screens load without errors", async ({ page }) => {
  for (const [path, heading] of [
    ["/thresholds", "Threshold lab"],
    ["/model", "Model and monitoring"],
    ["/experiments", "Experiments"],
  ] as const) {
    await page.goto(path);
    await expect(page.getByRole("heading", { level: 1, name: heading })).toBeVisible();
    await expect(page.getByRole("status").filter({ hasText: "Loading" })).toHaveCount(0, {
      timeout: 15_000,
    });
    await expect(page.getByRole("alert")).toHaveCount(0);
  }
});
