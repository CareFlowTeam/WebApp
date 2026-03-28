FROM node:20-alpine AS build

WORKDIR /app

ARG VITE_FLASK_BASE=
ARG VITE_FASTAPI_BASE=

ENV VITE_FLASK_BASE=${VITE_FLASK_BASE}
ENV VITE_FASTAPI_BASE=${VITE_FASTAPI_BASE}

COPY frontend/package*.json ./
RUN npm ci

COPY frontend .
RUN npm run build

FROM nginx:1.27-alpine

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80