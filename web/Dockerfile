# Console image: build the static files, then serve them with an unprivileged nginx.
# Build from the repository root: docker build -f web/Dockerfile .

FROM node:26.13.1-alpine AS build
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
# The experiments screen shows these generated results; they are never typed by hand.
COPY docs/results/experiments.json /app/docs/results/experiments.json
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.30.6-alpine
COPY web/nginx/default.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/web/dist /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=6 \
    CMD wget -qO- http://127.0.0.1:8080/healthz || exit 1
