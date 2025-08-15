import numpy as np
import functools
def ADB_solve(D):
    print(len(D))
    D = D - np.mean(D, axis=0, keepdims=True)

    
    eye = np.eye(len(D))

    norms = [np.linalg.norm(d) for d in D]

#    indices = np.argsort(norms)

#    D = D[indices]
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

        if len(permutation_so_far) == 9:
            PAD = D[permutation_so_far]
            PD = D[:len(permutation_so_far)]
            Q, residuals, rank, _singular_values = np.linalg.lstsq(PD, PAD)


            if np.abs(np.linalg.det(Q) + 1) < .001:
                return False

        if len(permutation_so_far) == len(D):
            PA = eye[permutation_so_far]
            permutations.append(PA)
            #rotations.append(Q)
            return False

        #return len(residuals) == 0 or (np.max(np.abs(residuals)) < 0.001).item()
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
    return permutations#, #rotations
def uADB_solve(D):
    eye = np.eye(len(D))

    @functools.cache
    def get_PDDPPT(N):
        P = eye[: N]
        PD = P @ D
        PDDP = np.linalg.inv(PD.T @ PD + .0001 * np.eye(3))
        return PD, PDDP @ PD.T



    def feasible(permutation_so_far):
        if ( np.abs(np.linalg.norm(D[len(permutation_so_far) - 1]) - np.linalg.norm(D[permutation_so_far[-1]])) > 0.001):
            return False

        PA = eye[permutation_so_far]
        PAD = PA @ D

        PD, PDDPPDT = get_PDDPPT(len(permutation_so_far))
        Q = PDDPPDT @ PAD

        apprx_PAD = PD @ Q

        residuals = PAD - apprx_PAD


        if np.abs(np.linalg.det(Q) + 1) < .001:
            return False

        if len(permutation_so_far) == len(D):
            permutations.append(PA)
            rotations.append(Q)
            return False
        try:
            if len(permutation_so_far) > 2:
                if np.max(np.abs(Q.T - np.linalg.inv(Q))) > .001:
                    return False
        except:
            pass

        return len(residuals) == 0 or (np.max(np.abs(residuals)) < 0.001).item()

    def recursive_permutations(permutation_so_far):
        for i in range(len(D)):
            if not i in permutation_so_far:
                continuation = permutation_so_far + [i]
                if feasible(continuation):
                    recursive_permutations(continuation)

    permutations = []
    rotations = []
    recursive_permutations([])
    return permutations#, rotations
def ADB_solve_2(D):
    """
    Given a matrix D of size N x M, find all pairs of A, B such that A is a permutation, B is a rotation, and AD = DB

    equivalently, given a set of points, find all rotations that permute the points
    """

    N = len(D)
    eye = np.eye(N)

    iterations = [0]

    def test(D, prefix):
        iterations[0] += 1

        D = D - np.mean(D, axis=0, keepdims=True)
        if (
            np.abs(np.linalg.norm(D[len(prefix) - 1]) - np.linalg.norm(D[prefix[-1]]))
            > 0.001
        ):
            return False
        if len(prefix) != len(set(prefix)):
            return False
        A = eye[prefix]
        targ = eye[: len(prefix)]
        transformed_A = A @ D
        transformed_identity = targ @ D
        Q = (
            np.linalg.inv(transformed_A.T @ transformed_A + np.eye(3) * 0.000001)
            @ transformed_A.T
            @ transformed_identity
        )

        if np.abs(np.linalg.det(Q) + 1) < 0.001:
            return False
        if len(prefix) == N:
            permutations.append(A)
            rotations.append(Q)
            return False
        return (
            np.max(np.abs(transformed_A.T - Q @ transformed_identity.T)) < 0.01
        ).item()

    def recursive_solve(D, prefix):
        for i in range(N):
            oo = prefix + [i]
            if test(D, oo):
                recursive_solve(D, oo)

    permutations = []
    rotations = []
    print("solving")
    recursive_solve(D, [])
    print("solved")
    print(iterations)
    return permutations#, rotations


