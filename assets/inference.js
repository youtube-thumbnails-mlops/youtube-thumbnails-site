
// Class Names mapped from Training Script indices
const CLASS_NAMES = [
    "Film & Animation",      // 0  (ID 1)
    "Autos & Vehicles",      // 1  (ID 2)
    "Music",                 // 2  (ID 10)
    "Pets & Animals",        // 3  (ID 15)
    "Sports",                // 4  (ID 17)
    "Travel & Events",       // 5  (ID 19)
    "Gaming",                // 6  (ID 20)
    "People & Blogs",        // 7  (ID 22)
    "Comedy",                // 8  (ID 23)
    "Entertainment",         // 9  (ID 24)
    "News & Politics",       // 10 (ID 25)
    "Howto & Style",         // 11 (ID 26)
    "Education",             // 12 (ID 27)
    "Science & Technology",  // 13 (ID 28)
    "Nonprofits & Activism", // 14 (ID 29)
    "Movies",                // 15 (ID 30)
    "Shows"                  // 16 (ID 43)
];

let session = null;

// Initialize ONNX Session
async function initModel() {
    try {
        console.log("⏳ Loading ONNX model...");
        // Option: set execution providers (wasm, webgl)
        session = await ort.InferenceSession.create('./assets/model.onnx', {
            executionProviders: ['wasm']
        });
        console.log("✅ Model loaded successfully!");
        // Fetch Metadata
        let versionText = "";
        try {
            const resp = await fetch('./assets/model_metadata.json');
            if (resp.ok) {
                const meta = await resp.json();
                versionText = ` (${meta.version})`;
            }
        } catch (e) {
            console.warn("Metadata load failed", e);
        }

        document.getElementById("model-status").innerHTML = `<i class="fas fa-check-circle" style="color: #00ff00;"></i> <span>Model Ready${versionText}</span>`;
        document.getElementById("model-status").classList.add("text-green-500");

        // Enable buttons
        const btns = document.querySelectorAll(".btn-predict");
        btns.forEach(b => {
            b.disabled = false;
            b.classList.add("btn-active"); // Optional visual cue
        });
    } catch (e) {
        console.error("❌ Failed to load model:", e);
        document.getElementById("model-status").innerHTML = `<i class="fas fa-times-circle" style="color: #ff0000;"></i> <span>Model Failed</span>`;
        document.getElementById("model-status").classList.add("text-red-500");
    }
}

// Preprocessing: Image -> Tensor
async function preprocess(imageElement) {
    const width = 224;
    const height = 224;

    // Draw to canvas to get data
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0, width, height);

    const imageData = ctx.getImageData(0, 0, width, height).data;

    // Float32 Array for Tensor (1, 3, 224, 224)
    // Normalize: (Pixel - Mean) / Std
    // Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]
    const floatArr = new Float32Array(1 * 3 * width * height);

    let i = 0; // pixel index
    for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
            let r = imageData[i * 4 + 0] / 255.0;
            let g = imageData[i * 4 + 1] / 255.0;
            let b = imageData[i * 4 + 2] / 255.0;

            // Normalize
            r = (r - 0.485) / 0.229;
            g = (g - 0.456) / 0.224;
            b = (b - 0.406) / 0.225;

            // CHW Layout: Red, then Green, then Blue
            // Index logic: c * (H*W) + h * W + w
            floatArr[0 * width * height + h * width + w] = r;
            floatArr[1 * width * height + h * width + w] = g;
            floatArr[2 * width * height + h * width + w] = b;

            i++;
        }
    }

    return new ort.Tensor('float32', floatArr, [1, 3, width, height]);
}

// Run Inference
async function runInference(imageElement) {
    if (!session) {
        alert("Model not loaded yet!");
        return;
    }

    try {
        const inputTensor = await preprocess(imageElement);

        // Input name from export: 'input'
        const feeds = { input: inputTensor };

        const start = performance.now();
        const results = await session.run(feeds);
        const end = performance.now();

        // Output name from export: 'output'
        const outputTensor = results.output;
        const data = outputTensor.data; // Float32Array of scores (logits)

        // Softmax (Optional, but good for confidence)
        const expData = data.map(x => Math.exp(x));
        const sumExp = expData.reduce((a, b) => a + b, 0);
        const probs = expData.map(x => x / sumExp);

        // Find Argmax
        let maxIdx = 0;
        let maxScore = -Infinity;
        for (let i = 0; i < probs.length; i++) {
            if (probs[i] > maxScore) {
                maxScore = probs[i];
                maxIdx = i;
            }
        }

        // Return Result
        return {
            category: CLASS_NAMES[maxIdx] || `Category ${maxIdx}`,
            confidence: maxScore,
            latency: (end - start).toFixed(2)
        };

    } catch (e) {
        console.error("Inference Error:", e);
        throw e;
    }
}
