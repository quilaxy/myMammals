let model;

// Log versi TensorFlow.js
console.log("TensorFlow.js Version:", tf.version.tfjs);

async function loadModel() {
    try {
        console.log("Loading model...");
        model = await tf.loadGraphModel('./tfjs_model/model.json');
        console.log("Model loaded successfully");

        // Log input shape model
        console.log("Model Input Shape:", model.inputs);
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

    // Preprocess the image
    const imgTensor = tf.browser.fromPixels(canvas)
        .resizeBilinear([177, 177]) // Resize ke ukuran input model
        .expandDims(0) // Tambahkan dimensi batch
        .toFloat()
        .div(255.0) // Normalisasi nilai piksel ke [0, 1]
        .sub(tf.tensor([0.485, 0.456, 0.406])) // Kurangi mean
        .div(tf.tensor([0.229, 0.224, 0.225])); // Bagi std

    console.log("Preprocessed Tensor:", imgTensor.arraySync());

    // Perform prediction
    const logits = await model.predict(imgTensor).data();
    console.log("Logits:", logits);

    // Apply softmax to convert logits to probabilities
    const probabilities = tf.softmax(tf.tensor(logits)).arraySync();
    console.log("Probabilities:", probabilities);

    // Define class labels
    const labels = ["Camel", "Koala", "Orangutan", "Snow Leopard", "Squirrel", "Water Buffalo", "Zebra"];

    // Sort probabilities in descending order
    const sortedIndices = Array.from(probabilities.keys()).sort((a, b) => probabilities[b] - probabilities[a]);

    // Highlight the top prediction
    const primaryIndex = sortedIndices[0];
    const resultTextElement = document.getElementById('result-text');
    if (resultTextElement) {
        resultTextElement.innerText = `${labels[primaryIndex]} (${(probabilities[primaryIndex] * 100).toFixed(2)}%)`;
    } else {
        console.error("Element with ID 'result-text' not found in the DOM.");
    }

    // Display the next top 6 classes and probabilities
    const resultContainer = document.getElementById('other-results');
    if (resultContainer) {
        resultContainer.innerHTML = ""; // Clear previous results
        sortedIndices.slice(1, 7).forEach((index) => {
            const className = labels[index];
            const probability = (probabilities[index] * 100).toFixed(2);
            const resultElement = document.createElement('p');
            resultElement.innerText = `${className} (${probability}%)`;
            resultContainer.appendChild(resultElement);
        });
    } else {
        console.error("Element with ID 'other-results' not found in the DOM.");
    }
}

// Ensure the model is loaded after the DOM is ready
document.addEventListener("DOMContentLoaded", loadModel);
