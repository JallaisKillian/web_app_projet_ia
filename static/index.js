const avisForm = document.getElementById("avisForm");

avisForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(avisForm);

    const response = await fetch("/predict", {method: "POST", body: formData});    
    if (response.ok) {
        const data = await response.json()

        showPrediction(data.prediction);
        avisForm.reset()
    } else {
        alert("Une erreur c'est produite");
    }
})

function showPrediction(prediction) {
    alert(`Prédiction : ${prediction}`)
}
