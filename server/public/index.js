console.log("hello from console");


const button = document.getElementById('button');
button.addEventListener('click', startDeliveringMedia);

let stream;
const connection = new RTCPeerConnection();

connection.addEventListener('track', mediaReceived);
connection.addEventListener('icecandidate', iceCandidate);

const video = document.getElementById('video');


async function startDeliveringMedia(){
    stream = await navigator.mediaDevices.getUserMedia({video:true});
    connection.addTrack(stream.getVideoTracks()[0], stream);

    // create offer:
    const offer = await connection.createOffer();
    console.log(offer)
    await connection.setLocalDescription(offer)
}

async function mediaReceived(event){
    video.srcObject = event.streams[0]
}
async function iceCandidate(event){

    if(!event.candidate){
        console.log("ice gathering finished.");

        const response = await fetch('/offer', {
            method: 'POST',
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(connection.localDescription)
        });


        const answer = await response.json();
        await connection.setRemoteDescription(answer);
    }

}