const canvas = document.getElementById('backgroundCanvas');
const ctx = canvas.getContext('2d');

let points = [];

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

function generateRandomPoints(count) {
    points = [];
    for (let i = 0; i < count; i++) {
        points.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            radius: Math.random() * 5 + 1,
            color: `rgba(${Math.random() * 255},${Math.random() * 255},${Math.random() * 255},0.5)`
        });
    }
}

function drawPoints() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    points.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, point.radius, 0, Math.PI * 2);
        ctx.fillStyle = point.color;
        ctx.fill();
    });
}

window.addEventListener('resize', () => {
    resizeCanvas();
    generateRandomPoints(100);
    console.log(points);
    drawPoints();
});

resizeCanvas();
generateRandomPoints(100);
drawPoints();