const canvas = document.getElementById("canvas"); // Retrieves the element of the canvas
const ctx = canvas.getContext("2d"); // Gets the 2D drawing context
ctx.fillStyle = "white"; // Sets the background color of the canvas
ctx.fillRect(0, 0, canvas.width, canvas.height); // Fills the background color of the canvas

let drawing = false; // Flag to determine if the mouse is pressed down or not

canvas.addEventListener("mousedown", () => (drawing = true)); // When mouse is clicked, drawing starts
canvas.addEventListener("mouseup", () => { // When the button is released
    drawing = false; // Sets the drawing flag as false
    ctx.beginPath(); // Resets the current path so the next line won’t connect to the previous one.
});

canvas.addEventListener("mousemove", draw); // Calls the draw function if the mouse is moving

// Function to draw in the canvas
function draw(e) {
    if (!drawing) {
        return; // Exits the function if it is not drawing
    }
    const rect = canvas.getBoundingClientRect(); // Gets the canvas position and size relative to the viewport
    const x = e.clientX - rect.left; // Converts mouse x-coordinates to canvas x-coordinates
    const y = e.clientY - rect.top; // Converts mouse y-coordinates to canvas y-coordinates

    ctx.lineWidth = 10; // Sets the line thickness
    ctx.lineCap = "round"; // Sets the line end caps
    ctx.strokeStyle = "black"; // Sets the line color

    ctx.lineTo(x, y); // Draws a line from the old (x, y) to the new (x, y)
    ctx.stroke(); // Renders the line
    ctx.beginPath(); // Resets the path 
    ctx.moveTo(x, y); // Moves the point to where the new lines will be drawn
}

// Function to clear the canvas
function clearCanvas() {
    ctx.fillStyle = "white"; // Sets the background color of the canvas
    ctx.fillRect(0, 0, canvas.width, canvas.height); // Fills the background color of the canvas
    document.getElementById("prediction").innerText = "Prediction: ?"; // Resets the prediction text
}

// A function that checks if the canvas is blank
function isCanvasBlank(canvas) {
    const blank = document.createElement("canvas"); // Creates a new canvas element
    blank.width = canvas.width; // Sets the new canvas's width to the original canvas
    blank.height = canvas.height; // Sets the new canvas's height to the original canvas

    const blankCtx = blank.getContext("2d"); // Gets the 2D drawing context
    blankCtx.fillStyle = "white"; // Sets the background color of the canvas
    blankCtx.fillRect(0, 0, blank.width, blank.height); // Fills the background color of the canvas

    return canvas.toDataURL() === blank.toDataURL(); // Compares the canvas to an empty canvas
}
  
// A function that uses the neural network to predict the drawed digit
async function predict() {
    if (isCanvasBlank(canvas)) {
        // If the canvas is blank
        alert("Please enter in a value!"); // Alerts the user to draw a value
        return; // Stops the function
    }

    const dataURL = canvas.toDataURL("image/png"); // Get the image as base64 PNG string

    const response = await fetch("/predict", {
        // Sends the image to the server’s /predict endpoint as JSON
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataURL }),
    });

    const result = await response.json(); // Waits for the server’s JSON response
    document.getElementById("prediction").innerText = `Prediction: ${result.prediction}`; // Updates the page to display the prediction result
}
