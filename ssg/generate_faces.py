import numpy as np
import json
from scipy.linalg import logm, expm
import itertools


def render_coob(polyhedron, axisN, depth):
    view = np.linalg.qr(np.random.randn(3, 3))[0]

    def loop(arr):
        return np.array(list(arr) + [arr[0]])
    mont = np.random.randn(3, 3)

    def fix(polygon):
        polygon = list(set(map(tuple, polygon)))
        yeet = np.array(polygon)
        yeet = yeet - np.mean(yeet, axis=0, keepdims=True)

        yeet = yeet @ mont

        yeet = np.atan2(*yeet.T)

        result= np.array(polygon)[np.argsort(yeet)]
        return result
    def gapcut(polygons, colors, normal, cutoff):
        gap = .010234
        out = []
        out_colors = []
        for polygon, color in zip(polygons, colors):
            polygon = fix(polygon)
            polygon = loop(polygon)
            p1 = []
            p2 = []
            cutoff += gap
            for i in range(len(polygon) - 1):
                dot1 = np.dot(polygon[i], normal)
                if dot1 == cutoff:
                    p1.append(polygon[i])
                elif dot1 > cutoff:
                    p1.append(polygon[i])
                    dot2 = np.dot(polygon[i + 1], normal)
                    if dot2 < cutoff:
                        t = (cutoff - dot2) / (dot1 - dot2)
                        p1.append((t * polygon[i] + (1 - t) * polygon[i + 1]))
                else:
                    dot2 = np.dot(polygon[i + 1], normal)
                    if dot2 > cutoff:
                        t = (cutoff - dot2) / (dot1 - dot2)
                        p1.append((t * polygon[i] + (1 - t) * polygon[i + 1]))
            cutoff -= 2 * gap
            for i in range(len(polygon) - 1):
                dot1 = np.dot(polygon[i], normal)
                if dot1 == cutoff:
                    p2.append(polygon[i])
                elif dot1 > cutoff:
                    dot2 = np.dot(polygon[i + 1], normal)
                    if dot2 < cutoff:
                        t = (cutoff - dot2) / (dot1 - dot2)
                        p2.append((t * polygon[i] + (1 - t) * polygon[i + 1]))
                else:
                    p2.append(polygon[i])
                    dot2 = np.dot(polygon[i + 1], normal)
                    if dot2 > cutoff:
                        t = (cutoff - dot2) / (dot1 - dot2)
                        p2.append((t * polygon[i] + (1 - t) * polygon[i + 1]))
            cutoff += gap

            if len(p1) >= 3:
                out.append(fix(p1))
                out_colors.append(color)
            if len(p2) >= 3:
                out.append(fix(p2))
                out_colors.append(color)

        return out, out_colors

    def ADB_solve(D):
        D = D - np.mean(D, axis=0, keepdims=True) 
        eye = np.eye(len(D))
        norms = [np.linalg.norm(d) for d in D]
        norms = np.array(norms)
        prefixes = []
        max_dots = 5
        for L in range(len(D) + 1):
            PD = D[:L]
            prefixes.append(PD[-1:] @ PD[:max_dots].T)
        def feasible(permutation_so_far):
            if ( np.abs(norms[len(permutation_so_far) - 1] - norms[permutation_so_far[-1]]) > 0.001):
                return False
            if not np.max(np.abs(D[permutation_so_far[-1:]] @ D[permutation_so_far[:max_dots]].T - prefixes[len(permutation_so_far)])) < .0001:
                return False
            if len(permutation_so_far) == 6:
                PAD = D[permutation_so_far]
                PD = D[:len(permutation_so_far)]
                Q, residuals, rank, _singular_values = np.linalg.lstsq(PD, PAD)
                if np.abs(np.linalg.det(Q) + 1) < .001:
                    return False
            if len(permutation_so_far) == len(D):
                PAD = D[permutation_so_far]
                PD = D[:len(permutation_so_far)]
                Q, residuals, rank, _singular_values = np.linalg.lstsq(PD, PAD)
                if np.abs(np.linalg.det(Q) + 1) < .001:
                    return False

                PA = eye[permutation_so_far]
                permutations.append(PA)
                rotations.append(Q)
                return False
            return True

        def recursive_permutations(permutation_so_far):
            for i in range(len(D)):
                if not i in permutation_so_far:
                    continuation = permutation_so_far + [i]
                    if feasible(continuation):
                        recursive_permutations(continuation)

        permutations = []
        rotations = []
        recursive_permutations([])
        return permutations, rotations

    if polyhedron == "cube":
        coob = np.array(list(itertools.product(*[[-1, 1]] * 3)))

        face_pts = 4


    if polyhedron == "octahedron":
        coob = 2 *np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, -1, 0], [-1, 0, 0], [0, 0, -1]])
        face_pts = 3

    if polyhedron == "tetrahedron":
        coob = np.array([[1, -1, -1], [-1, 1, -1], [1, 1, 1], [-1, -1, 1]])
        face_pts = 3

    if polyhedron == "icosohedron":
        phi = (1 + np.sqrt(5)) / 2
        coob = [[0, 1, phi], [phi, 0, 1], [1, phi, 0], [0, 1, -phi], [0, -1, phi], [0, -1, -phi], 
                [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]]

        face_pts = 3

    if polyhedron == "dodecahedron":
        phi = (1 + np.sqrt(5)) / 2
        coob = [[0, 1, phi], [phi, 0, 1], [1, phi, 0], [0, 1, -phi], [0, -1, phi], [0, -1, -phi], 
                [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]]
        face_pts = 3
        symmetries, rotations = ADB_solve(coob)
        faces = set()

        for sym in symmetries:
            face = tuple(sorted(map(tuple, (sym @ coob) [:face_pts])))
            c = np.mean(face, axis=0)
            face = tuple(map(tuple, c + (face - c) * .99))

            faces.add(face)

        coob = np.array([np.mean(face, axis=0) for face in faces])
        coob = np.array(sorted(coob.tolist(), key=lambda a: a[0] + a[1]))
        face_pts=5

    symmetries, rotations = ADB_solve(coob)
    print(len(symmetries))
    faces = set()

    for sym in symmetries:
        face = tuple(sorted(map(tuple, (sym @ coob) [:face_pts])))
        c = np.mean(face, axis=0)
        face = tuple(map(tuple, c + (face - c) * .99))
        faces.add(face)

    faces = np.array(list(faces))
    colors = [ "red", "green", "blue", "white", "yellow", 
              "orange", "purple", "pink", "gray", "brown", "black", "cyan", "darkgreen", "darkgray", "tan", "lime", "navy", "darkcyan", "gold"]

    cut_axes = [[v.real for v in np.linalg.eig(ro).eigenvectors.T if np.all(v == v.real)][0] for ro in rotations]

    axis = cut_axes[axisN ]

    cfaces = faces
    ccolors = colors
    cutoff = depth

    axes = []
    for ro in rotations[:]:
        yeee = ro @ axis
        if any(np.allclose(tt, yeee, rtol=.01) for tt in axes):
            continue
        axes.append(yeee)
        cfaces, ccolors = gapcut(cfaces, ccolors, yeee, cutoff)

    arr = np.concatenate(cfaces)
    global_perms, uu = ADB_solve(arr)

    cue = (arr @ axis > cutoff)
    slice_ = arr[cue]
    sliceperm, Q = ADB_solve(slice_)
    slicemove = np.eye(len(arr))
    halfslice = slicemove[cue]
    halfslice[:, cue] = sliceperm[1]
    slicemove[cue] = halfslice


    def np2js(arr):
        real_arr = np.block([[arr.real, arr.imag], [-arr.imag, arr.real]])
        return json.dumps(real_arr.tolist())


    sticker_coords = arr
    sticker_colors = 127 + 127 * np.round(sticker_coords / 1.9)

    view = np.linalg.qr(np.random.randn(3, 3))[0]

    with open(f"{polyhedron}{depth}{axisN}output.html", "w") as static_site:
        static_site.write(
            f"""
        <!DOCTYPE html>
        <html>
        <body>
        <canvas id="canvas" height=500 width=500></canvas>
        <script src="live.js"></script>
        <script>
        const ctx = document.getElementById("canvas").getContext('2d');

        var state = {np2js(np.eye(len(sticker_coords)))}
        const coords = {np2js(sticker_coords @ view * 100 + 255)}
        """)

        static_site.write("""
        const toSparse = (M) => {
        const sparse = [];
        for (let i = 0; i < M.length; i++) {
            for (let j = 0; j < M[i].length; j++) {
                if (Math.abs(M[i][j]) >= 0.0000001) {
                    sparse.push({row: i, col: j, val: M[i][j]});
                }
            }
        }
        return sparse;
    };

    let mul = (A, B) => {
        const m = A.length;
        const n = B[0].length;
        const p = B.length;
        const sparseA = toSparse(A);
        const sparseB = toSparse(B);

        const bByRow = {};
        for (const {row, col, val} of sparseB) {
            if (!bByRow[row]) bByRow[row] = [];
            bByRow[row].push({col, val});
        }
        const products = {};
        for (const {row: i, col: k, val: aVal} of sparseA) {
            if (bByRow[k]) {
                for (const {col: j, val: bVal} of bByRow[k]) {
                    const key = `${i},${j}`;
                    products[key] = (products[key] || 0) + aVal * bVal;
                }
            }
        }
        const result = Array(m).fill().map(() => Array(n).fill(0));
        for (const [key, val] of Object.entries(products)) {
            const [i, j] = key.split(',').map(Number);
            result[i][j] = val;
        }

        return result;
    };
        var moves = [state]
        document.addEventListener("keypress", (event) => {
        """
        )
        for i, generator in enumerate([slicemove, global_perms[5], global_perms[9]]):
            static_site.write(
                f"""
            if (event.key == {i}) {{
                moves = (new Array(10).fill( {np2js(expm(.1 * logm(generator)))})).concat( moves);
            }}
            """
            )
        static_site.write(
            """
        });
        step = () => {
            if (!moves.length) {
                requestAnimationFrame(step)
                return;
            }
            state = mul(state, moves.pop());
            const locations = mul(state, coords);
            ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            drawlist = [
        """
        )
        i = 0
        for face, color in zip(cfaces, ccolors):
            static_site.write("[")

            for point in face:
            
                static_site.write(
                    f"""
                    [...locations[{i}] ,"{color}"], 
                """
                )
                i += 1
            static_site.write("],")
        static_site.write(
            """
            ].sort((a, b) => a[0][0] + a[1][0] + a[2][0] -b[0][0] - b[1][0] - b[2][0])
            for (var face of drawlist){
                ctx.beginPath()
                for ( var pt of face) {
                ctx.lineTo(pt[1], pt[2]);
                
                ctx.fillStyle=pt[6]

                }
                ctx.lineTo(face[0][1], face[0][2]);
                
                ctx.fill()
                ctx.stroke()
            }
            requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
        </script>
        </body>
        </html>
        """
        )

#render_coob("dodecahedron", 3, .8)
render_coob("icosohedron", 5, 1.7)
