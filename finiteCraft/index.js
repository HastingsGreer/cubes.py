let dragged = null;
let last = [-1000, -1000];
let clickstart = [-100, -100];
let clickoffset = [-100, -100];
let lastButton = -100;

function set_adjusted_position(el, x, y) {
  el.style.left = x - clickoffset[0] + "px";
  el.style.top = y - clickoffset[1] + "px";
}

function createItem(matrix) {
  const el = document.createElement("div");
  el.className = "item";
  el.innerHTML = getCube(matrix);
  el.dataset.permutationmatrix = JSON.stringify(matrix);
  el.onpointerdown = (e) => {
    clickstart = [e.clientX, e.clientY];
    var el_rect = el.getBoundingClientRect();
    clickoffset = [e.clientX - el_rect.left + 3, e.clientY - el_rect.top + 3];
    dragged = { matrix, el: el.cloneNode(true) };
    dragged.el.style.position = "absolute";
    dragged.el.style.zIndex= "10000";
    set_adjusted_position(dragged.el, e.clientX, e.clientY);
    document.body.appendChild(dragged.el);
    if (el.parentElement.id === "workspace" && e.button === 0) el.remove();
    lastButton = e.button;
    e.preventDefault();
  };
  el.oncontextmenu = (e) => e.preventDefault();
	el.ontouchstart = (e) => {
  e.preventDefault();
};
  return el;
}

function renderSidebar() {
  sidebar.innerHTML = "";
  discovered.forEach((t) => sidebar.appendChild(createItem(t)));
}

document.onpointermove = (e) => {
  if (dragged) {
    set_adjusted_position(dragged.el, e.clientX, e.clientY);
	  e.preventDefault();
  }
};

function maybe_add_to_sidebar(combined) {
  if (
    !discovered.some((old_matrix) => {
      return JSON.stringify(old_matrix) === JSON.stringify(combined);
    })
  ) {
    discovered.push(combined);
    discovered.sort(
      (a, b) =>
        numeric.sum(numeric.abs(numeric.sub(a, numeric.identity(54)))) -
        numeric.sum(numeric.abs(numeric.sub(b, numeric.identity(54)))),
    );
  }
  renderSidebar();
}

function getHit(target) {
  const combined = numeric.dot(
    JSON.parse(target.dataset.permutationmatrix),
    dragged.matrix,
  );
  maybe_add_to_sidebar(combined);
  const combined_item = createItem(combined);
  combined_item.style.left = target.style.left;
  combined_item.style.top = target.style.top;
  doAnimation(
    combined_item,
    JSON.parse(target.dataset.permutationmatrix),
    combined,
  );
  target.remove();
  workspace.appendChild(combined_item);
}

document.onpointerup = (e) => {
  if (!dragged) return;
  dragged.el.remove();

  var drop_x = e.clientX,
    drop_y = e.clientY;

  if (Math.abs(drop_x - clickstart[0]) + Math.abs(drop_y - clickstart[1]) < 8) {
    drop_x = last[0];
    drop_y = last[1];
    const target = [...workspace.querySelectorAll(".item")].find((el) => {
      const r = el.getBoundingClientRect();
      return (
        drop_x >= r.left &&
        drop_x <= r.right &&
        drop_y >= r.top &&
        drop_y <= r.bottom
      );
    });

    if (target) {
      getHit(target);
      last = [drop_x, drop_y];
    }

    if (e.clientX < workspace.offsetWidth && lastButton === 0) {
      const dropped_item = createItem(dragged.matrix);
      set_adjusted_position(dropped_item, e.clientX, e.clientY);
      workspace.appendChild(dropped_item);
    }
    dragged = null;
    return;
  }

  const target = [...workspace.querySelectorAll(".item")].find((el) => {
    const r = el.getBoundingClientRect();
    return (
      drop_x >= r.left &&
      drop_x <= r.right &&
      drop_y >= r.top &&
      drop_y <= r.bottom
    );
  });

  if (target) {
    getHit(target);
    last = [drop_x, drop_y];
  } else {
    if (drop_x < workspace.offsetWidth) {
      const dropped_item = createItem(dragged.matrix);
      set_adjusted_position(dropped_item, e.clientX, e.clientY);
      workspace.appendChild(dropped_item);
    }
  }
  dragged = null;
};

function doAnimation(item, old_matrix, new_matrix) {
  var oldLocations = numeric.dot(old_matrix, coords);
  var newLocations = numeric.dot(new_matrix, coords);
  var stickers = [...item.firstElementChild.children];
  stickers.forEach((sticker) => {
    sticker.style.transition = "left 0.3s ease-out, top 0.3s ease-out";
  });
  stickers.forEach((sticker, i) => {
    sticker.style.left = oldLocations[i][1] + "px";
    sticker.style.top = oldLocations[i][2] + "px";
    sticker.style.zIndex = Math.round(oldLocations[i][0]);
  });
  setTimeout(() => {
    stickers.forEach((sticker, i) => {
      sticker.style.left = newLocations[i][1] + "px";
      sticker.style.top = newLocations[i][2] + "px";
      sticker.style.zIndex = Math.round(newLocations[i][0]);
    });
  }, 20);
}

renderSidebar();
document.oncontextmenu = (e) => e.preventDefault();
