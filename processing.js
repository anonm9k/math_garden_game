
var model; // model.json
// load model
async function loadModel() {
    model = await tf.loadGraphModel('TFJS/model.json')
}

function predictImage() {
    
    // 1. load image
    let image = cv.imread(canvas); // getting image
    // 2. change color to greyscale
    cv.cvtColor(image, image, cv.COLOR_RGBA2GRAY, 0);
    cv.threshold(image, image, 175, 255, cv.THRESH_BINARY);
    // 3. get image contours
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    
    cv.findContours(image, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
    // 4. cut the image by contour
    let cnt = contours.get(0);
    let rect = cv.boundingRect(cnt);
    image = image.roi(rect);
    // 5. resizing image(20 pixels)
    var height = image.rows;
    var width = image.cols;

    if (height > width) {
        height = 20;
        const scaleFactor = image.rows / height;
        width = Math.round(image.cols / scaleFactor);
    } else {
        width = 20;
        const scaleFactor = image.cols / width;
        height = Math.round(image.rows / scaleFactor);
    }

    let newSize = new cv.Size(width, height);
    cv.resize(image, image, newSize, 0, 0, cv.INTER_AREA)
    // 6. Add padding each side
    const LEFT = Math.ceil(4 + (20 - width) / 2);
    const RIGHT = Math.floor(4 + (20 - width) / 2);
    const TOP = Math.ceil(4 + (20 - height) / 2);
    const BOTTOM = Math.floor(4 + (20 - height) / 2);

    const BLACK = new cv.Scalar(0, 0, 0, 0);// black color
    cv.copyMakeBorder(image, image, TOP, BOTTOM, LEFT, RIGHT, cv.BORDER_CONSTANT, BLACK);

    // 7. Calculate center of mass
    // get countour of resized image
    cv.findContours(image, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
    cnt = contours.get(0);
    const Moments = cv.moments(cnt, false);// moment to calculate mass
    // from documentation
    const cx = Moments.m10 / Moments.m00;
    const cy = Moments.m01 / Moments.m00;
    // 8. shiting the image in the center by mass
    const X_SHIFT = Math.round(image.cols / 2.0 - cx);
    const Y_SHIFT = Math.round(image.rows / 2.0 - cy);

    newSize = new cv.Size(image.cols, image.rows);
    const M = cv.matFromArray(2, 3, cv.CV_64FC1, [1, 0, X_SHIFT, 0, 1, Y_SHIFT]);
    cv.warpAffine(image, image, M, newSize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, BLACK);
    // 9. Changing pixel values(from 1-255 to 1-0)
    let pixelValues = image.data;

    pixelValues = Float32Array.from(pixelValues);// transforming dtype to float

    pixelValues = pixelValues.map(function (item) {
        return item / 255.0;// using map to apply fucntion to all elements in js array
    });

    // 10. making a tensor for prediction
    const X = tf.tensor([pixelValues]);

    // console.log(`Shape of Tensor: ${X.shape}`); (1, 784)
    // console.log(`dtype of Tensor: ${X.dtype}`); (float32)

    // 11. Predict value
    const result = model.predict(X);
    result.print();
    const output = result.dataSync()[0];


    // console.log(tf.memory());
    
    // Testing Only (delete later)
    // const outputCanvas = document.createElement('CANVAS');
    // cv.imshow(outputCanvas, image);
    // document.body.appendChild(outputCanvas);

    // Cleanup
    image.delete();
    contours.delete();
    cnt.delete();
    hierarchy.delete();
    M.delete();
    X.dispose(); // cleaning memory
    result.dispose();

    return output;

}