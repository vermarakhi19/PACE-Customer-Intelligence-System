// main.js — sidebar toggle + theme toggle (dark is default; toggles to light)
document.addEventListener("DOMContentLoaded", function () {
  const sidebar = document.getElementById("sidebar");
  const sidebarToggle = document.getElementById("sidebarToggle");
  if (sidebarToggle) {
    sidebarToggle.addEventListener("click", () => sidebar.classList.toggle("open"));
  }

  // Desktop sidebar collapse (persisted across pages/sessions)
  const collapseToggle = document.getElementById("collapseToggle");
  if (collapseToggle && sidebar) {
    if (localStorage.getItem("rp-sidebar-collapsed") === "1") {
      sidebar.classList.add("collapsed");
    }
    collapseToggle.addEventListener("click", () => {
      const collapsed = sidebar.classList.toggle("collapsed");
      localStorage.setItem("rp-sidebar-collapsed", collapsed ? "1" : "0");
    });
  }

  const themeToggle = document.getElementById("themeToggle");
  const html = document.documentElement;
  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const isDark = html.getAttribute("data-theme") !== "light";
      html.setAttribute("data-theme", isDark ? "light" : "dark");
      themeToggle.innerHTML = isDark
        ? '<i class="fa-solid fa-moon"></i>'
        : '<i class="fa-solid fa-sun"></i>';
      themeToggle.title = isDark ? "Toggle dark mode" : "Toggle light mode";
    });
  }
});

// Shared Chart.js palette matching the Aurora Intelligence design tokens
const CHART_COLORS = ["#8B6BFF", "#2DD4E8", "#34D399", "#FBBF24", "#FB7185", "#6D4FE0", "#A78BFA"];

function isLightTheme() { return document.documentElement.getAttribute("data-theme") === "light"; }

Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.color = isLightTheme() ? "#565F78" : "#A6AEC4";
Chart.defaults.borderColor = isLightTheme() ? "rgba(227,230,240,0.8)" : "rgba(38,44,61,0.8)";
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.boxWidth = 8;
Chart.defaults.plugins.legend.labels.padding = 14;
Chart.defaults.elements.line.borderWidth = 2.5;
Chart.defaults.elements.bar.borderRadius = 6;
Chart.defaults.elements.point.radius = 3;
Chart.defaults.elements.point.hoverRadius = 5;
