import numpy as np
import json
from scipy.linalg import logm, expm
import itertools

def rigid_perms(D, prefix=[]):
    return (
        sum(
            (
                rigid_perms(D, prefix + [i])
                for i in range(len(D))
                if (
                    D[prefix + [i]] @ D[i] == D[: len(prefix) + 1] @ D[len(prefix)]
                ).all()
            ),
            [],
        )
        if len(prefix) < len(D)
        else [np.eye(len(D))[prefix]]
    )

def np2js(arr):
    return json.dumps(arr.tolist())

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

with open("output.js", "w") as static_site:
    static_site.write(
        f"""

    let mul = (A, B) => A.map((row, i) => B[0].map((_, j) =>
        row.reduce((acc, _, n) => acc + A[i][n] * B[n][j], 0)))

    window.coords = {np2js(sticker_coords @ view * 14 + 35)}
    window.discovered = [
    """
    )
    for i, generator in enumerate([move, global_perms[15], global_perms[10]]):
        static_site.write(
            f"""
            {np2js(generator)},
        """
        )
    static_site.write(
        """
    ];
    function getCube(state) {
        const locations = mul(state, coords);
        return  ` <div style='width: 93px; height: 93px; position: relative'>
    """
    )
    for i, color in enumerate(sticker_colors):
        static_site.write(
            f"""
            <div style='
                position: absolute; 
                left: ${{locations[{i}][1]}}px; 
                top: ${{locations[{i}][2]}}px; 
                z-index: ${{Math.round(10 * locations[{i}][0])}}; 
                background-color: rgb({color[0]} {color[1]} {color[2]});
                border: 1px solid black;
                width: 20px;
                height: 20px;
                border-radius: 10px;
                transition: 
                "left 0.3s ease-out, top 0.3s ease-out";
            '>
            </div>
        """
        )
    static_site.write("</div>`}")
