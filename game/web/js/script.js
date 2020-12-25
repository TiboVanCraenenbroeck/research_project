let domGameGrid, domGameGridCell, domTotScore, domShapeQueue, domBtnRender;
let domLoaded = false, render = true;

eel.expose(startFront);
function startFront(data) {
  loadDom();
  domBtnRender.addEventListener('click', changeRenderBtn);
}
const changeRenderBtn = function(){
  // Change the bool
  eel.change_render_state()
};
eel.expose(changedRender)
function changedRender(data){
  // Set the var to the backend-var
  render = data
  // Change the text in the btn
  if(render){
    domBtnRender.innerHTML = "On";
  }else{
    domBtnRender.innerHTML = "Off";
  }
};
const changeGrid = async function (grid) {
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
const createShape = async function (shape) {
  let output = "";
  for (const s of shape) {
    for (const c of s) {
      output += `<div class="c-queue-shape--shape c-cell--color" data-shape-nr="${c}"></div>`;
    }
  }
  return output;
};
const showShapesQueue = async function (shapesQueue) {
  for (const [indexShape, shapeQueue] of shapesQueue.entries()) {
    let standardShape = [
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
    ];
    for (let indexRow = 0; indexRow < shapeQueue.shape.length; indexRow++) {
      shape = shapeQueue.shape[indexRow];
      for (let indexCol = 0; indexCol < shape.length; indexCol++) {
        standardShape[indexRow][indexCol] = shape[indexCol];
      }
    }
    domShapeQueue[indexShape].innerHTML = await createShape(standardShape);
  }
};
const changeScore = async function (score) {
  domTotScore.innerHTML = score;
};
const changeGame = async function (grid, score, shapeQueue) {
  showShapesQueue(shapeQueue);
  changeGrid(grid);
  await changeScore(score);
};

/* EEL */
eel.expose(changeGameGrid);
function changeGameGrid(grid, score, shape_queue) {
  changeGame(grid, score, shape_queue);
}

const loadExposes = function () {
  eel.startBackend("System Started!");
};
const loadDom = function () {
  domGameGrid = document.querySelector(".js-game-grid");
  domGameGridCell = document.querySelectorAll(".js-game-grid--cell");
  domTotScore = document.querySelector(".js-tot-score");
  domShapeQueue = document.querySelectorAll(".js-game-shapes-queue--shape");
  domBtnRender = document.querySelector(".js-render");
  domLoaded = true;
};
document.addEventListener("DOMContentLoaded", function () {
  console.log("JS loaded!");
  loadExposes();
});
