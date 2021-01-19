$(document).ready(function(){
/* 트랙 바 */
        const slideValue = document.querySelector("span");
        const inputSlider = document.querySelector("input");
        
        inputSlider.oninput = (()=>{
            let value = inputSlider.value;
            slideValue.textContent = value;
            slideValue.style.left = (value/255*100*0.75) + "%";
            slideValue.classList.add("show");
        });
        
        /* 비디오 버튼 */
        var video = document.querySelector("#videoElement");
        var onoff = document.getElementById('btn');
        
        onoff.onclick = function(){
            navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUsermedia || navigator.mozGetUserMedia || navigato.msGetUserMedia || navigator.oGetUserMedia;

            if(navigator.getUserMedia){
                  navigator.getUserMedia({video:true}, handleVideo, videoError)      
            }

            function handleVideo(stream){
                video.srcObject = stream;
                video.play();
            }

            function videoError(e){}
            
            onoff.onclick = null;
        }
});