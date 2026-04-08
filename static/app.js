function parseDashboardPayload() {
    const payloadElement = document.getElementById("dashboardPayload");
    if (!payloadElement) {
        return null;
    }

    try {
        return JSON.parse(payloadElement.textContent);
    } catch (error) {
        return null;
    }
}

function revealOnScroll() {
    const items = document.querySelectorAll("[data-reveal]");
    if (!items.length) {
        return;
    }

    if (!("IntersectionObserver" in window)) {
        items.forEach((item) => item.classList.add("is-visible"));
        return;
    }

    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("is-visible");
                    observer.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.18 }
    );

    items.forEach((item) => observer.observe(item));
}

function enableLoadingState() {
    const form = document.getElementById("predictionForm");
    const submitButton = document.querySelector("[data-submit-button]");

    if (!form || !submitButton) {
        return;
    }

    form.addEventListener("submit", () => {
        if (!form.checkValidity()) {
            return;
        }

        form.classList.add("is-loading");
        form.setAttribute("aria-busy", "true");
        submitButton.setAttribute("disabled", "disabled");
    });
}

function enableDemoFill() {
    const triggers = document.querySelectorAll("[data-demo-preset]");
    if (!triggers.length) {
        return;
    }

    triggers.forEach((trigger) => {
        trigger.addEventListener("click", () => {
            let presetValues = null;
            try {
                presetValues = JSON.parse(trigger.dataset.demoPreset || "{}");
            } catch (error) {
                presetValues = null;
            }

            if (!presetValues) {
                return;
            }

            const inputs = document.querySelectorAll("input[name]");
            inputs.forEach((input) => {
                if (Object.prototype.hasOwnProperty.call(presetValues, input.name)) {
                    input.value = presetValues[input.name];
                }
            });

            triggers.forEach((button) => button.classList.remove("is-active"));
            trigger.classList.add("is-active");

            const firstInput = document.querySelector("input[name]");
            if (firstInput) {
                firstInput.focus();
            }
        });
    });
}

function scrollToResults() {
    const resultPanel = document.getElementById("resultPanel");
    if (!resultPanel || resultPanel.dataset.hasResult !== "true") {
        return;
    }

    window.requestAnimationFrame(() => {
        resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    });
}

function buildChartOptions(overrides = {}) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 900,
            easing: "easeOutQuart",
        },
        plugins: {
            legend: {
                labels: {
                    color: "#4e6157",
                    usePointStyle: true,
                    boxWidth: 10,
                    padding: 18,
                    font: {
                        family: "DM Sans",
                        size: 12,
                    },
                },
            },
            tooltip: {
                backgroundColor: "rgba(24, 49, 38, 0.95)",
                titleColor: "#ffffff",
                bodyColor: "#f5f1e8",
                padding: 12,
                displayColors: false,
            },
        },
        ...overrides,
    };
}

function renderDashboardCharts() {
    const payload = parseDashboardPayload();
    if (!payload || typeof Chart === "undefined") {
        return;
    }

    const centerTextPlugin = {
        id: "centerTextPlugin",
        afterDraw(chart, args, pluginOptions) {
            if (chart.config.type !== "doughnut") {
                return;
            }

            const meta = chart.getDatasetMeta(0);
            if (!meta || !meta.data || !meta.data.length) {
                return;
            }

            const { ctx } = chart;
            const centerPoint = meta.data[0];
            const x = centerPoint.x;
            const y = centerPoint.y;

            ctx.save();
            ctx.textAlign = "center";
            ctx.fillStyle = "#183126";
            ctx.font = "700 34px Sora";
            ctx.fillText(pluginOptions.value || "", x, y - 6);
            ctx.fillStyle = "#62756b";
            ctx.font = "500 12px 'DM Sans'";
            ctx.fillText(pluginOptions.label || "", x, y + 18);
            ctx.restore();
        },
    };

    if (!Chart.registry.plugins.get("centerTextPlugin")) {
        Chart.register(centerTextPlugin);
    }

    const confidenceCanvas = document.getElementById("confidenceChart");
    if (confidenceCanvas) {
        new Chart(confidenceCanvas, {
            type: "doughnut",
            data: {
                labels: ["Confidence", "Uncertainty"],
                datasets: [
                    {
                        data: [payload.confidence, payload.uncertainty],
                        backgroundColor: ["#2d7a4c", "rgba(24, 49, 38, 0.10)"],
                        borderWidth: 0,
                        hoverOffset: 4,
                    },
                ],
            },
            options: buildChartOptions({
                cutout: "74%",
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label(context) {
                                return `${context.label}: ${context.raw}%`;
                            },
                        },
                    },
                    centerTextPlugin: {
                        value: `${payload.confidence}%`,
                        label: "confidence",
                    },
                },
            }),
        });
    }

    const radarCanvas = document.getElementById("radarChart");
    if (radarCanvas && payload.radar && payload.radar.labels.length) {
        new Chart(radarCanvas, {
            type: "radar",
            data: {
                labels: payload.radar.labels,
                datasets: [
                    {
                        label: "Health alignment score",
                        data: payload.radar.values,
                        backgroundColor: "rgba(45, 122, 76, 0.14)",
                        borderColor: "#2d7a4c",
                        borderWidth: 2,
                        pointBackgroundColor: "#2d7a4c",
                        pointBorderColor: "#ffffff",
                        pointRadius: 3,
                    },
                ],
            },
            options: buildChartOptions({
                scales: {
                    r: {
                        min: 0,
                        max: 100,
                        ticks: {
                            stepSize: 20,
                            backdropColor: "transparent",
                            color: "#819489",
                            font: {
                                family: "DM Sans",
                                size: 11,
                            },
                        },
                        pointLabels: {
                            color: "#183126",
                            font: {
                                family: "DM Sans",
                                size: 12,
                                weight: "700",
                            },
                        },
                        grid: {
                            color: "rgba(24, 49, 38, 0.10)",
                        },
                        angleLines: {
                            color: "rgba(24, 49, 38, 0.08)",
                        },
                    },
                },
                plugins: {
                    legend: { display: false },
                },
            }),
        });
    }

    const importanceCanvas = document.getElementById("importanceChart");
    if (importanceCanvas && payload.importance && payload.importance.labels.length) {
        new Chart(importanceCanvas, {
            type: "bar",
            data: {
                labels: payload.importance.labels,
                datasets: [
                    {
                        label: "Driver strength",
                        data: payload.importance.values,
                        backgroundColor: payload.importance.colors,
                        borderRadius: 12,
                        borderSkipped: false,
                    },
                ],
            },
            options: buildChartOptions({
                indexAxis: "y",
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label(context) {
                                return `Driver strength: ${context.raw}%`;
                            },
                        },
                    },
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            color: "#819489",
                            callback(value) {
                                return `${value}%`;
                            },
                            font: {
                                family: "DM Sans",
                            },
                        },
                        grid: {
                            color: "rgba(24, 49, 38, 0.08)",
                        },
                    },
                    y: {
                        ticks: {
                            color: "#183126",
                            font: {
                                family: "DM Sans",
                                weight: "700",
                            },
                        },
                        grid: {
                            display: false,
                        },
                    },
                },
            }),
        });
    }
}

document.addEventListener("DOMContentLoaded", () => {
    revealOnScroll();
    enableLoadingState();
    enableDemoFill();
    scrollToResults();
    renderDashboardCharts();
});
