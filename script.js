let model;

// Log versi TensorFlow.js
console.log("TensorFlow.js Version:", tf.version.tfjs);

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

    // Preprocess the image
    const imgTensor = tf.browser.fromPixels(canvas)
        .resizeBilinear([177, 177]) // Resize ke ukuran input model
        .expandDims(0) // Tambahkan dimensi batch
        .toFloat()
        .div(255.0) // Normalisasi nilai piksel ke [0, 1]
        .sub(tf.tensor([0.4804, 0.4785, 0.4361])) // Kurangi mean
        .div(tf.tensor([0.2149, 0.2122, 0.2146])); // Bagi std

    console.log("Preprocessed Tensor:", imgTensor.arraySync());

    // // Perform prediction
    // const logits = await model.predict(imgTensor).data();
    // console.log("Logits:", logits);

    // // Apply softmax to convert logits to probabilities
    // const probabilities = tf.softmax(tf.tensor(logits)).arraySync();
    // console.log("Probabilities:", probabilities);

    const probabilities = await model.predict(imgTensor).data();
    console.log("Probabilities:", probabilities);

    // Define class labels
    const labels = ["Camel", "Dolphin", "Koala", "Orangutan", "Snow Leopard", "Water Buffalo", "Zebra"];

    // Sort probabilities in descending order
    const sortedIndices = Array.from(probabilities.keys()).sort((a, b) => probabilities[b] - probabilities[a]);

    // Display all classes and probabilities as bars
    const resultsContainer = document.getElementById('results');
    resultsContainer.innerHTML = ""; // Clear previous results

    sortedIndices.forEach((index, rank) => {
        const className = labels[index];
        const probability = (probabilities[index] * 100).toFixed(2);

        // Create a container for the bar
        const progressContainer = document.createElement('div');
        progressContainer.classList.add('progress-container');

        // Add class label
        const labelElement = document.createElement('div');
        labelElement.classList.add('class-label');
        if (rank === 0) {
            labelElement.classList.add('highlight'); // Highlight top prediction
        }
        labelElement.innerText = className;
        progressContainer.appendChild(labelElement);

        // Add progress bar
        const progressBarContainer = document.createElement('div');
        progressBarContainer.classList.add('progress');
        const progressBar = document.createElement('div');
        progressBar.classList.add('progress-bar');
        progressBar.style.width = `${probability}%`;
        progressBar.innerText = `${probability}%`;

        // Add progress bar to the container
        progressBarContainer.appendChild(progressBar);
        progressContainer.appendChild(progressBarContainer);

        // Add container to results
        resultsContainer.appendChild(progressContainer);
    });
}

document.addEventListener("DOMContentLoaded", loadModel);
