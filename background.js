let canvas = document.getElementById('backgroundCanvas');
let ctx = canvas.getContext('2d');
const content = document.getElementById('content');

function loadMarkdown(file) {
    fetch(file)
        .then(response => response.text())
        .then(text => {
            const converter = new showdown.Converter();
            content.innerHTML = converter.makeHtml(text);
        });
}

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

function drawPoints(points) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const minDimension = Math.min(canvas.width, canvas.height);
    points.forEach(point => {
        ctx.beginPath();
        ctx.arc(
            point[0] * canvas.width,
            point[1] * canvas.height,
            0.005 * minDimension,
            0,
            Math.PI * 2
        );
        let color = ``;
        if (point[2] === 0) {
            color = `rgba(0,0,255,0.3)`;
        } else if (point[2] === 1) {
            color = `rgba(255,0,0,0.3)`;
        } else if (point[2] === 2) {
            color = `rgba(0,0,0,0.3)`;
        }
        ctx.fillStyle = color;
        ctx.fill();
    });
}

function addBoundaryToPoints(points, boundary) {
    // points has the format [[x, y, label], ...]
    // boundary has the format [[x, y], ...]
    // add the boundary to the points with label 2
    // first delete all points with label 2
    points = points.filter(point => point[2] !== 2);
    return points.concat(boundary.map(point => [point[0], point[1], 2]));
}

async function main() {
    let pyodide = await loadPyodide();
    await pyodide.loadPackage("numpy");
    // Read the python script from a file as plain text
    let script = await fetch('nn.py').then(response => response.text());
    pyodide.runPython(script);
    let generate = pyodide.globals.get("generate");
    let initialize_model = pyodide.globals.get("initialize_model");
    let step = pyodide.globals.get("step");
    initialize_model();
    // After a training sesh on one data, regenerate the data with the same model
    for (let n = 0; n < 999999; n++) {
        let points = await generate();
        resizeCanvas();
        drawPoints(points);
        await new Promise(resolve => setTimeout(resolve, 100));
        // Try 100 steps
        for (let i = 0; i < 400; i++) {
            // Get the new points
            boundary = await step();
            // short delay to allow the canvas to render after 250 steps
            // Add the boundary to the points
            points = addBoundaryToPoints(points, boundary);
            drawPoints(points);
            await new Promise(resolve => setTimeout(resolve, 25));
        }
    }
}

main();

window.addEventListener('resize', () => {
    resizeCanvas();
    drawPoints(points);
});