eel.expose(changeGameGrid);
function changeGameGrid(data){
    alert(data)
}
/* const changeGameGrid = function (data) {
  alert(data)
}; */

const loadExposes = function () {
    changeGameGrid("Blob")
    eel.my_python_function("Dit is een test")
};

document.addEventListener("DOMContentLoaded", function () {
  console.log("Loaded!");
  loadExposes();
});
