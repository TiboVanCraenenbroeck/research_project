let domGameGridCell;
let domLoaded = false;

eel.expose(startFront);
function startFront(data) {
  console.log(data);
  loadDom();
}
eel.expose(changeGameGrid);
function changeGameGrid(grid, score, shape_queue) {
  console.log(grid);
  grid_json = JSON.parse(grid);
  if (domLoaded) {
    let divNr = 0;
    for (const row of grid_json) {
      for (const col of row) {
        domGameGridCell[divNr].setAttribute("data-shape-nr", `${col}`);
        divNr++;
      }
    }
  }
}

const loadExposes = function () {
  eel.startBackend("System Started!");
};
const loadDom = function () {
  domGameGridCell = document.querySelectorAll(".js-game-grid--cell");
  domLoaded = true;
};
document.addEventListener("DOMContentLoaded", function () {
  console.log("JS loaded!");
  loadExposes();
});
