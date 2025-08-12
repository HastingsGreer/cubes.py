import numpy as np
import json
from scipy.linalg import logm, expm
import itertools


def rigid_perms(D, prefix=[]):
    return (
        sum(
            (
                rigid_perms(D, prefix + [i])
                for i in {*range(len(D))} - {*prefix}
                if (D[prefix + [i]] @ D[i] == D[: len(prefix) + 1] @ D[len(prefix)]).all()
            ),
            [],
        )
        if len(prefix) < len(D)
        else [np.eye(len(D))[prefix]]
    )


def np2js(arr):
    real_arr = np.block([[arr.real, arr.imag], [-arr.imag, arr.real]])
    return json.dumps(real_arr.tolist())


sticker_coords = np.array(
    [
        s
        for s in itertools.product(range(-2, 3), repeat=3)
        if (np.abs(s) == 2).sum() == 1
    ]
)
sticker_colors = 127 + 127 * np.round(sticker_coords / 2)
global_perms = rigid_perms(sticker_coords)
slice_perms = rigid_perms(sticker_coords[:21])
move = np.eye(len(sticker_coords))
move[:21, :21] = slice_perms[3]
view = np.linalg.qr(np.random.randn(3, 3))[0]

with open("output.html", "w") as static_site:
    static_site.write(f"""
    <!DOCTYPE html>
    <html>
    <body>
    <canvas id="canvas" height=500 width=500></canvas>
    <script src="live.js"></script>
    <script>
    const ctx = document.getElementById("canvas").getContext('2d');

    let mul = (A, B) => A.map((row, i) => B[0].map((_, j) =>
        row.reduce((acc, _, n) => acc + A[i][n] * B[n][j], 0)))

    var state = {np2js(np.eye(len(sticker_coords)))}
    const coords = {np2js(sticker_coords @ view * 90 + 255)}
    var moves = [state]
    document.addEventListener("keypress", (event) => {{
    """)
    for i, generator in enumerate([move, global_perms[15], global_perms[9]]):
        static_site.write( f"""
        if (event.key == {i}) {{
            moves = (new Array(10).fill( {np2js(expm(.1 * logm(generator)))})).concat( moves);
        }}
        """)
    static_site.write( """
    });
    step = () => {
        if (!moves.length) {
            requestAnimationFrame(step)
            return;
        }
        state = mul(state, moves.pop());
        const locations = mul(state, coords);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawlist = [
    """)
    for i, color in enumerate(sticker_colors):
        static_site.write( f"""
            [...locations[{i}], 'rgb({color[0]} {color[1]} {color[2]})'], 
        """)
    static_site.write("""
        ].sort((a, b) => a[0]-b[0])
        for (var elem of drawlist){
            ctx.beginPath()
            ctx.arc(elem[1], elem[2], 40, 0, 2 * Math.PI)
            ctx.fillStyle=elem[6]
            ctx.fill()
            ctx.stroke()
        }
        requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
    </script>
    </body>
    </html>
    """)
