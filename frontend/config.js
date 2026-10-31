// Local development only (e.g. `python -m http.server 8080 -d frontend`).
// Inside the container nginx generates /config.js from BACKEND_URL instead.
window.FINSIGHT_CONFIG = { backendUrl: "http://localhost:8000", mlflowUrl: "http://localhost:5000",
  prometheusUrl: "http://localhost:9090", grafanaUrl: "http://localhost:3000",
  alertmanagerUrl: "http://localhost:9093" };
