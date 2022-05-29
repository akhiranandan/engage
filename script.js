//Getting files that are uploading
const block = document.getElementById('file')

//Loading Various models that we are using to make face detection asynchronously
Promise.all([faceapi.nets.faceRecognitionNet.loadFromUri('/models'), faceapi.nets.faceLandmark68Net.loadFromUri('/models'), faceapi.nets.ssdMobilenetv1.loadFromUri('/models')]).then(Begin)

async function Begin() 
{

  const elements = document.createElement('div')
  elements.style.position = 'relative'
  document.body.append(elements)
  const Criminal_detector = await Criminal_Images_Load()
  const Match = new faceapi.FaceMatcher(Criminal_detector, 0.6)

  let Picture
  let canvas

  block.addEventListener('change', async () => 
  {

    if (Picture) Picture.remove()
    if (canvas) canvas.remove()

    //Taking the file which we have uploaded and converting to image element
    Picture = await faceapi.bufferToImage(block.files[0])
    elements.append(Picture)
    canvas = faceapi.createCanvasFromMedia(Picture)
    elements.append(canvas)
    const Area_of_display = {height: Picture.height, width: Picture.width}
    faceapi.matchDimensions(canvas, Area_of_display)
    const identity = await faceapi.detectAllFaces(Picture).withFaceLandmarks().withFaceDescriptors()
    const newIdentity = faceapi.resizeResults(identity, Area_of_display)
    const detections = newIdentity.map(d => Match.findBestMatch(d.descriptor))

    detections.forEach((result, i) => 
    {
      const box = newIdentity[i].detection.box
      const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
      drawBox.draw(canvas)
    })
  })

}

function Criminal_Images_Load() {
  //Names of the folders which contain Criminal images
  const Criminals = ['Abu Mohammad al-Adnani', 'Dawood Ibrahim', 'Gurmeet Ram Rahim Singh', 'Haji Mastan', 'Ilyas Kashmiri', 'Masood Azhar', 'Osama bin Laden', 'Syed Salahuddin' , 'Veerappan']
  return Promise.all(
    Criminals.map(async label => 
      {
      //initialising an empty array
      const descriptions = []
      
      for (let i = 1; i <= 2; i++) 
      {
        const img = await faceapi.fetchImage(`https://raw.githubusercontent.com/akhiranandan/Labeled_images/main/Criminal_images/${label}/${i}.jpg`)
        const identity = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
        descriptions.push(identity.descriptor)
      }
          
      return new faceapi.LabeledFaceDescriptors(label, descriptions)
      })
  )
}
