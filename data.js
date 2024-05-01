const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

const TRAIN_IMAGES_DIR = './data/train';
const TEST_IMAGES_DIR = './data/test';

function loadImages(dataDir) {
  const images = [];
  const labels = [];
  var folders = fs.readdirSync(dataDir)
  folders.map((folder, index) => {
    var files = fs.readdirSync(path.join(dataDir,folder));
    for (let i = 0; i < files.length; i++) { 

      var filePath = path.join(dataDir, folder, files[i]);
      
      var buffer = fs.readFileSync(filePath);
      var imageTensor = tf.node.decodeImage(buffer)
        .resizeNearestNeighbor([96,96])
        .mean(2)
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims()
        .expandDims(-1);
      images.push(imageTensor);
      labels.push(index);
    }
  });
  console.log('Labels are');
  console.log(labels);
  return [images, labels];
}

/** Helper class to handle loading training and test data. */
class DogDataset {
  constructor() {
    this.trainData = [];
    this.testData = [];
  }

  /** Loads training and test data. */
  loadData() {
    console.log('Loading images...');
    this.trainData = loadImages(TRAIN_IMAGES_DIR);
    this.testData = loadImages(TEST_IMAGES_DIR);
    console.log('Images loaded successfully.')
  }

  getTrainData() {
    return {
      
      images: tf.concat(this.trainData[0]),
      labels: tf.oneHot(tf.tensor1d(this.trainData[1], 'int32'), 5).toFloat() // here 5 is class
      
    }
  }

  getTestData() {
    return {
      images: tf.concat(this.testData[0]),
      labels: tf.oneHot(tf.tensor1d(this.testData[1], 'int32'), 5).toFloat()
    }
  }
}

module.exports = new DogDataset();
console.log('All done.')
