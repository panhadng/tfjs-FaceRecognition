const imageUpload = document.getElementById("imageUpload");

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
]).then(start);

async function start() {
  const container = document.createElement("div");
  container.style.position = "relative";
  document.body.append(container);
  const labeledFaceDescriptors = await loadLabeledImages();
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
  let image;
  let canvas;
  document.body.append("Loaded");
  imageUpload.addEventListener("change", async () => {
    if (image) image.remove();
    if (canvas) canvas.remove();
    image = await faceapi.bufferToImage(imageUpload.files[0]);
    container.append(image);
    canvas = faceapi.createCanvasFromMedia(image);
    container.append(canvas);
    const displaySize = { width: image.width, height: image.height };
    faceapi.matchDimensions(canvas, displaySize);
    const detections = await faceapi
      .detectAllFaces(image)
      .withFaceLandmarks()
      .withFaceDescriptors();
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    const results = resizedDetections.map((d) =>
      faceMatcher.findBestMatch(d.descriptor)
    );
    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, {
        label: result.toString(),
      });
      drawBox.draw(canvas);
    });
  });
}

async function loadLabeledImages() {
  const images = {
    "Black Widow": [
      "./labeled_images/Black Widow/1.jpg",
      "./labeled_images/Black Widow/2.jpg",
    ],
    "Captain America": [
      "./labeled_images/Captain America/1.jpg",
      "./labeled_images/Captain America/2.jpg",
    ],
    "Captain Marvel": [
      "./labeled_images/Captain Marvel/1.jpg",
      "./labeled_images/Captain Marvel/2.jpg",
    ],
    Hawkeye: [
      "./labeled_images/Hawkeye/1.jpg",
      "./labeled_images/Hawkeye/2.jpg",
    ],
    "Jim Rhodes": [
      "./labeled_images/Jim Rhodes/1.jpg",
      "./labeled_images/Jim Rhodes/2.jpg",
    ],
    Thor: ["./labeled_images/Thor/1.jpg", "./labeled_images/Thor/2.jpg"],
    "Tony Stark": [
      "./labeled_images/Tony Stark/1.jpg",
      "./labeled_images/Tony Stark/2.jpg",
    ],
  };

  labels = [];
  for (var key in images) {
    for (const link of images[key]) {
      labels.push({
        name: key,
        img: link,
      });
    }
  }
  const labeledFaceDescriptors = await Promise.all(
    labels.map(async (label, index) => {
      const img = await faceapi.fetchImage(label["img"]);
      const fullFaceDescription = await faceapi
        .detectSingleFace(img)
        .withFaceLandmarks()
        .withFaceDescriptor();
      if (!fullFaceDescription) {
        throw new Error(`no faces detected for ${label["name"]}`);
      }
      const faceDescriptors = [fullFaceDescription.descriptor];
      return new faceapi.LabeledFaceDescriptors(label["name"], faceDescriptors);
    })
  );
  return labeledFaceDescriptors;
}
