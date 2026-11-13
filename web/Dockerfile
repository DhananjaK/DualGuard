# DualGuard dashboard image: build the React app, serve it with unprivileged nginx.

# Stage 1 – build (Node is only needed here, not in the final image)
FROM node:22-alpine AS build
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2 – serve static files + proxy /api to the API container
FROM nginxinc/nginx-unprivileged:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /web/dist /usr/share/nginx/html
EXPOSE 8080
