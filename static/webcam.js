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