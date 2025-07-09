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

    ctx.lineWidth = 15;
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

let model;
async function loadModel() {
    model = await tf.loadLayersModel(
        "https://storage.googleapis.com/tfjs-models/tfjs/mnist/model.json"
    );
    console.log("Model loaded");
}

async function predict() {
    if (!model) {
        alert("Model not loaded yet!");
        return;
    }

    const smallCanvas = document.createElement("canvas");
    smallCanvas.width = 28;
    smallCanvas.height = 28;
    const smallCtx = smallCanvas.getContext("2d");

    smallCtx.drawImage(canvas, 0, 0, 28, 28);

    const imgData = smallCtx.getImageData(0, 0, 28, 28);
    const data = [];
    for (let i = 0; i < imgData.data.length; i += 4) {
        const avg = (imgData.data[i] + imgData.data[i + 1] + imgData.data[i + 2]) / 3;
        data.push((255 - avg) / 255);
    }

    const input = tf.tensor(data, [1, 28, 28, 1]);
    const prediction = model.predict(input);
    const predictedDigit = prediction.argMax(1).dataSync()[0];

    document.getElementById("prediction").innerText = `Prediction: ${predictedDigit}`;

    input.dispose();
    prediction.dispose();
}

loadModel();
