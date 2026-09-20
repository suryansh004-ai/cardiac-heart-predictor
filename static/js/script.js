/**
 * Heart Disease Prediction System - Client Script
 * Handles demonstration profile presets and form validation.
 */

document.addEventListener("DOMContentLoaded", function () {
    const predictionForm = document.getElementById("prediction-form");
    const btnLoadLowRisk = document.getElementById("btn-load-low-risk");
    const btnLoadHighRisk = document.getElementById("btn-load-high-risk");
    const btnResetForm = document.getElementById("btn-reset-form");

    // Demonstration Profile 1: Lower Cardiovascular Risk Sample (Model Outcome: Class 0)
    const sampleLowRisk = {
        age: 55,
        sex: "1",        // Male
        cp: "0",         // Typical angina
        trestbps: 130,   // BP
        chol: 230,       // Cholesterol
        fbs: "0",        // Fasting blood sugar <= 120
        restecg: "0",    // Normal resting ECG
        thalach: 120,    // Heart rate
        exang: "1",      // Exercise angina
        oldpeak: 2.0,    // ST depression
        slope: "1",      // Flat slope
        ca: "2",         // 2 vessels
        thal: "3"        // Reversible defect
    };

    // Demonstration Profile 2: Higher Likelihood of Risk Sample (Model Outcome: Class 1)
    const sampleHighRisk = {
        age: 63,
        sex: "1",        // Male
        cp: "3",         // Asymptomatic
        trestbps: 145,   // Elevated BP (Stage 2)
        chol: 233,       // Serum cholesterol
        fbs: "1",        // Fasting blood sugar > 120
        restecg: "0",    // Normal resting ECG
        thalach: 150,    // Peak heart rate
        exang: "0",      // No exercise angina
        oldpeak: 2.3,    // Significant ST depression
        slope: "0",      // Upsloping
        ca: "0",         // 0 vessels
        thal: "1"        // Fixed defect
    };

    // Helper function to populate form fields
    function fillForm(data) {
        if (!predictionForm) return;
        Object.keys(data).forEach(fieldId => {
            const field = document.getElementById(fieldId);
            if (field) {
                field.value = data[fieldId];
                field.classList.remove("is-invalid");
            }
        });
    }

    // Attach button events if on the prediction page
    if (btnLoadLowRisk) {
        btnLoadLowRisk.addEventListener("click", function () {
            fillForm(sampleLowRisk);
        });
    }

    if (btnLoadHighRisk) {
        btnLoadHighRisk.addEventListener("click", function () {
            fillForm(sampleHighRisk);
        });
    }

    if (btnResetForm) {
        btnResetForm.addEventListener("click", function () {
            if (predictionForm) {
                predictionForm.reset();
                const fields = predictionForm.querySelectorAll(".form-control, .form-select");
                fields.forEach(f => f.classList.remove("is-invalid"));
            }
        });
    }

    // Client-side validation on form submit
    if (predictionForm) {
        predictionForm.addEventListener("submit", function (event) {
            let isValid = true;
            const requiredFields = predictionForm.querySelectorAll("[required]");

            requiredFields.forEach(field => {
                if (!field.value || field.value.trim() === "") {
                    isValid = false;
                    field.classList.add("is-invalid");
                } else {
                    field.classList.remove("is-invalid");
                }
            });

            // Specific range checks
            const ageInput = document.getElementById("age");
            if (ageInput && (parseFloat(ageInput.value) < 1 || parseFloat(ageInput.value) > 120)) {
                isValid = false;
                ageInput.classList.add("is-invalid");
            }

            const bpInput = document.getElementById("trestbps");
            if (bpInput && (parseFloat(bpInput.value) < 50 || parseFloat(bpInput.value) > 260)) {
                isValid = false;
                bpInput.classList.add("is-invalid");
            }

            const cholInput = document.getElementById("chol");
            if (cholInput && (parseFloat(cholInput.value) < 80 || parseFloat(cholInput.value) > 700)) {
                isValid = false;
                cholInput.classList.add("is-invalid");
            }

            const hrInput = document.getElementById("thalach");
            if (hrInput && (parseFloat(hrInput.value) < 40 || parseFloat(hrInput.value) > 250)) {
                isValid = false;
                hrInput.classList.add("is-invalid");
            }

            if (!isValid) {
                event.preventDefault();
                alert("Please check the form for invalid or missing values before submitting.");
            }
        });
    }
});
