FROM node:22-alpine
ARG OPENCODE_VERSION=1.18.5
RUN npm install -g "opencode-ai@${OPENCODE_VERSION}"
RUN adduser -D -u 10001 agent
USER agent
WORKDIR /work
# The runtime configuration is mounted read-only by compose.  This image has no
# application source, credentials or Docker socket.
EXPOSE 4096
CMD ["opencode", "serve", "--hostname", "0.0.0.0", "--port", "4096"]
