let domGameGridCell, domTotScore;
let domLoaded = false;

eel.expose(startFront);
function startFront(data) {
  loadDom();
}
const changeGrid = async function(grid){
  const gridJson = JSON.parse(grid);
  if (domLoaded) {
    let divNr = 0;
    for (const row of gridJson) {
      for (const col of row) {
        domGameGridCell[divNr].setAttribute("data-shape-nr", `${col}`);
        divNr++;
      }
    }
  }
};
const showShapesQueue = async function(shapesQueue){
  /* const shapesQueueJson = JSON.parse(shapesQueue) */
  for (const shape of shapesQueue) {
    console.log(shape);
  }
};
const changeScore = async function(score){
  domTotScore.innerHTML = score;
};
const changeGame = async function(grid, score, shapeQueue){
  showShapesQueue(shapeQueue)
  changeGrid(grid)
  await changeScore(score)
};
eel.expose(changeGameGrid);
function changeGameGrid(grid, score, shape_queue) {
  changeGame(grid, score, shape_queue)
}

const loadExposes = function () {
  eel.startBackend("System Started!");
};
const loadDom = function () {
  domGameGridCell = document.querySelectorAll(".js-game-grid--cell");
  domTotScore = document.querySelector(".js-tot-score");
  domLoaded = true;
};
document.addEventListener("DOMContentLoaded", function () {
  console.log("JS loaded!");
  loadExposes();
});
