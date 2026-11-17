import { describe, expect, it } from "vitest";

import { lab } from "../test/fixtures";
import { curvePoints, numberAt } from "./experiments";
import { age, logOdds, money, percent, utc } from "./format";
import { operatingPoint, thresholdLabel } from "./thresholds";
import { buildWaterfall } from "./waterfall";

describe("buildWaterfall", () => {
  const contributions = [
    { feature: "a", contribution: 1.5 },
    { feature: "b", contribution: -0.5 },
    { feature: "c", contribution: 3 },
    { feature: "d", contribution: 0.25 },
  ];

  it("walks from the baseline to the margin", () => {
    const steps = buildWaterfall(-6, contributions, 2);

    expect(steps.map((s) => s.label)).toEqual([
      "Baseline",
      "c",
      "a",
      "2 other features",
      "This transaction",
    ]);
    expect(steps[1]).toMatchObject({ start: -6, end: -3, kind: "up" });
    expect(steps[3]).toMatchObject({ start: -1.5, end: -1.75, kind: "down", value: -0.25 });
    expect(steps.at(-1)).toMatchObject({ start: 0, end: -1.75, kind: "total" });
  });

  it("adds no 'other' bar when every feature is shown", () => {
    const steps = buildWaterfall(0, contributions, 8);

    expect(steps).toHaveLength(contributions.length + 2);
    expect(steps.at(-1)?.end).toBeCloseTo(4.25);
  });
});

describe("operatingPoint", () => {
  it("reads the released pair", () => {
    const point = operatingPoint(lab, 1, 2);

    expect(point).toMatchObject({
      tReview: 0.02,
      tBlock: 0.5,
      costPer1000: 25.5,
      alertsPerDay: 15,
      blocksPerDay: 6,
      isReleased: true,
      withinCapacity: true,
    });
    expect(point.precision).toBeCloseTo(0.6);
  });

  it("raises a block threshold below the review threshold", () => {
    expect(operatingPoint(lab, 2, 0)).toMatchObject({ tBlock: 0.5, isReleased: false });
  });

  it("flags pairs over capacity and rejects unknown ones", () => {
    expect(operatingPoint(lab, 0, 3).withinCapacity).toBe(false);
    expect(() => operatingPoint(lab, 9, 9)).toThrow(RangeError);
  });

  it("labels the never sentinel", () => {
    expect(thresholdLabel(1.01)).toBe("never");
    expect(thresholdLabel(0.02)).toBe("0.0200");
  });
});

describe("formatting", () => {
  it("formats money, shares, log-odds and times", () => {
    expect(money(1234.5)).toBe("$1,234.50");
    expect(percent(0.1234)).toBe("12.3%");
    expect(percent(null)).toBe("n/a");
    expect(logOdds(-0.5)).toBe("−0.50");
    expect(logOdds(2)).toBe("+2.00");
    expect(utc("2020-08-14T03:12:44Z")).toBe("2020-08-14 03:12 UTC");
  });

  it("states ages in the largest whole unit", () => {
    const now = new Date("2020-01-02T00:00:00Z");
    expect(age("2020-01-01T23:59:30Z", now)).toBe("30 s");
    expect(age("2020-01-01T23:00:00Z", now)).toBe("1 h");
    expect(age("2020-01-01T23:15:00Z", now)).toBe("45 min");
    expect(age("2020-01-01T20:00:00Z", now)).toBe("4 h");
    expect(age("2019-12-30T00:00:00Z", now)).toBe("3 d");
  });
});

describe("numberAt", () => {
  it("reads nested numbers and nothing else", () => {
    const doc = { a: { b: 1.5, c: "x", d: Number.NaN } };
    expect(numberAt(doc, "a", "b")).toBe(1.5);
    expect(numberAt(doc, "a", "c")).toBeNull();
    expect(numberAt(doc, "a", "d")).toBeNull();
    expect(numberAt(doc, "missing", "b")).toBeNull();
    expect(numberAt(null, "a")).toBeNull();
  });

  it("pairs curve points", () => {
    expect(
      curvePoints({
        arm: "A",
        description: "",
        summary: {},
        pr_curve: { recall: [0, 1], precision: [1] },
      }),
    ).toEqual([
      { recall: 0, precision: 1 },
      { recall: 1, precision: Number.NaN },
    ]);
  });
});
