let model;
let class1Images = [];
let class2Images = [];
let classNames = [];

// Load MobileNet (pre-trained model)
async function loadModel() {
    model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    document.getElementById('status').innerText = "Model Loaded!";
}
loadModel();

// Function to convert images into tensors
function imageToTensor(img) {
    return tf.tidy(() => {
        let tensor = tf.browser.fromPixels(img)
            .resizeNearestNeighbor([224, 224]) // Resize to MobileNet input size
            .toFloat()
            .expandDims(); // Add batch dimension
        return tensor;
    });
}

// Handle file uploads
document.getElementById('class1-files').addEventListener('change', (event) => {
    class1Images = Array.from(event.target.files);
    classNames[0] = document.getElementById('class1-name').value || "Class 1";
});

document.getElementById('class2-files').addEventListener('change', (event) => {
    class2Images = Array.from(event.target.files);
    classNames[1] = document.getElementById('class2-name').value || "Class 2";
});

// Train Model
document.getElementById('train-btn').addEventListener('click', async () => {
    if (class1Images.length === 0 || class2Images.length === 0) {
        alert("Please upload images for both classes.");
        return;
    }

    document.getElementById('status').innerText = "Training model...";

    let xs = [];
    let ys = [];

    // Process class 1 images
    for (let imgFile of class1Images) {
        let img = await loadImage(imgFile);
        xs.push(imageToTensor(img));
        ys.push(0);
    }

    // Process class 2 images
    for (let imgFile of class2Images) {
        let img = await loadImage(imgFile);
        xs.push(imageToTensor(img));
        ys.push(1);
    }

    // Convert to tensors
    let xTrain = tf.stack(xs);
    let yTrain = tf.tensor1d(ys, 'int32');

    // Create a simple model
    const newModel = tf.sequential();
    newModel.add(tf.layers.flatten({ inputShape: [224, 224, 3] }));
    newModel.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    newModel.add(tf.layers.dense({ units: 2, activation: 'softmax' }));
    newModel.compile({
        optimizer: 'adam',
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy']
    });

    // Train the model
    await newModel.fit(xTrain, yTrain, {
        epochs: 5,
        batchSize: 8,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss}`);
            }
        }
    });

    document.getElementById('status').innerText = "Model trained!";
    model = newModel;
});

// Load images as HTML elements
function loadImage(file) {
    return new Promise((resolve) => {
        let img = new Image();
        img.src = URL.createObjectURL(file);
        img.onload = () => resolve(img);
    });
}

// Predict new image
document.getElementById('predict-btn').addEventListener('click', async () => {
    let file = document.getElementById('test-image').files[0];
    if (!file) {
        alert("Please upload an image for prediction.");
        return;
    }

    let img = await loadImage(file);
    let inputTensor = imageToTensor(img);
    let prediction = model.predict(inputTensor);
    let classIndex = prediction.argMax(1).dataSync()[0];

    document.getElementById('prediction').innerText = `Prediction: ${classNames[classIndex]}`;
});
