let model;

async function loadModel() {
    try {
        console.log("Loading model...");
        model = await tf.loadGraphModel('./tfjs_model/model.json'); 
        console.log("Model loaded successfully");
    } catch (error) {
        console.error("Error loading model:", error);
    }
}

function processImage(event) {
    const file = event.target.files[0];
    if (!file) return;

    const img = new Image();
    const reader = new FileReader();

    reader.onload = function (e) {
        img.src = e.target.result;
    };

    img.onload = function () {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        const size = 177; // Model expects 177x177 input
        canvas.width = size;
        canvas.height = size;

        const aspectRatio = img.width / img.height;
        let sx, sy, sw, sh;
        if (aspectRatio > 1) {
            sw = img.height;
            sh = img.height;
            sx = (img.width - sw) / 2;
            sy = 0;
        } else {
            sw = img.width;
            sh = img.width;
            sx = 0;
            sy = (img.height - sh) / 2;
        }

        ctx.drawImage(img, sx, sy, sw, sh, 0, 0, size, size);

        const resultImage = document.getElementById('result-image');
        resultImage.src = canvas.toDataURL('image/jpeg');
        resultImage.style.visibility = 'visible';

        classifyImage(canvas);
    };

    reader.readAsDataURL(file);
}

async function classifyImage(canvas) {
    if (!model) {
        alert("Model not loaded yet!");
        return;
    }

    // Preprocess the image to match model input
    const imgTensor = tf.browser.fromPixels(canvas)
        .resizeBilinear([177, 177]) // Resize to 177x177
        .expandDims(0) // Add batch dimension
        .toFloat()
        .div(255.0); // Normalize pixel values to [0, 1]

    console.log("Preprocessed Tensor:", imgTensor.arraySync());

    // Perform prediction
    const rawPredictions = await model.predict(imgTensor).data();
    const predictions = tf.softmax(tf.tensor(rawPredictions)).arraySync(); // Convert logits to probabilities

    // Define class labels
    const labels = ["Camel", "Koala", "Orangutan", "Snow Leopard", "Squirrel", "Water Buffalo", "Zebra"];

    // Get top prediction
    const sortedIndices = Array.from(predictions.keys()).sort((a, b) => predictions[b] - predictions[a]);
    const primaryIndex = sortedIndices[0];
    document.getElementById('result-text').innerText = `${labels[primaryIndex]} (${(predictions[primaryIndex] * 100).toFixed(2)}%)`;

    // Display top 5 predictions
    for (let i = 1; i <= 5; i++) {
        const index = sortedIndices[i];
        const resultElement = document.getElementById(`result${i + 1}`);
        resultElement.innerText = `${labels[index]} (${(predictions[index] * 100).toFixed(2)}%)`;
        resultElement.style.visibility = 'visible';
    }
}

document.addEventListener("DOMContentLoaded", loadModel);
