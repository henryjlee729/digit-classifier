const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;

canvas.addEventListener("mousedown", () => (drawing = true));
canvas.addEventListener("mouseup", () => {
    drawing = false;
    ctx.beginPath();
});

canvas.addEventListener("mousemove", draw);

function draw(e) {
    if (!drawing) {
        return;
    }
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.lineWidth = 5;
    ctx.lineCap = "round";
    ctx.strokeStyle = "black";

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function clearCanvas() {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById("prediction").innerText = "Prediction: ?";
}

function isCanvasBlank(canvas) {
    const blank = document.createElement("canvas");
    blank.width = canvas.width;
    blank.height = canvas.height;

    const blankCtx = blank.getContext("2d");
    blankCtx.fillStyle = "white"; // fill blank canvas with white
    blankCtx.fillRect(0, 0, blank.width, blank.height);

    return canvas.toDataURL() === blank.toDataURL();
}
  
async function predict() {
    if (isCanvasBlank(canvas)) {
        alert("Please enter in a value!");
        return;
    }
      
    const dataURL = canvas.toDataURL("image/png"); // Get the image as base64 PNG string

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataURL }),
    });

    const result = await response.json();
    document.getElementById("prediction").innerText = `Prediction: ${result.prediction}`;
}
