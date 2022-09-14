import cupy as np
import cudf as pd
from numba import cuda
import math, time
import argparse


@cuda.jit
def edge_triangle(in_col, dst_col, out_col, deg_mat, dst_list, src_dict):
    i = cuda.grid(1)
    if i < in_col.size: # boundary guard
        x = in_col[i]
        xnei = src_dict[x]
        y = dst_col[i]
        ynei = src_dict[y]
        xi=0
        yi=0
        while xi<deg_mat[x] and yi<deg_mat[y]:
            if dst_list[xnei+xi]==dst_list[ynei+yi]:# triangle xyz
                out_col[i] = out_col[i] + 1
                xi += 1
                yi += 1
            elif dst_list[xnei+xi]<dst_list[ynei+yi]:
                xi += 1
            else:
                yi += 1


@cuda.jit
def full_graphlet(in_col, out_col, neigh, deg_mat, dst_list, src_dict):
    i = cuda.grid(1)
    if i < in_col.size: # boundary guard
        x = in_col[i]
        xnei = src_dict[x]
        nx = 0
        while nx < deg_mat[x]:
            y = dst_list[xnei+nx]
            if y >= x:
                break
            nn = 0
            ynei = src_dict[y]
            ny = 0
            while ny < deg_mat[y]:
                z = dst_list[ynei+ny]
                if z >= y:
                    break
                tmp = 0
                while tmp < deg_mat[x]:
                    if dst_list[xnei+tmp] == z:# triangle xyz
                        neigh[xnei+nn]=z
                        nn += 1
                    elif dst_list[xnei+tmp] > z:
                        break
                    tmp += 1
                ny += 1
            tmpi = 0
            while tmpi < nn:
                z = neigh[xnei+tmpi]
                znei = src_dict[z]
                tmpi += 1
                tmpj = tmpi
                while tmpj < nn:
                    zz = neigh[xnei+tmpj]
                    tmp = 0
                    while tmp < deg_mat[z]:
                        if dst_list[znei+tmp] == zz:
                            out_col[x] = out_col[x] + 1
                            out_col[y] = out_col[y] + 1
                            out_col[z] = out_col[z] + 1
                            out_col[zz] = out_col[zz] + 1
                        elif dst_list[znei+tmp] > zz:
                            break
                        tmp += 1
                    tmpj += 1
            nx += 1


@cuda.jit
def equation_system(in_col, out_col, deg_mat, dst_list, src_dict, edge_tri, c4_list):
    i = cuda.grid(1)
    if i < in_col.size: # boundary guard
        x = in_col[i]
        xnei = src_dict[x]
        mybool = True
        tmpi = 0
        f_12_14=0
        f_10_13=0
        f_13_14=0
        f_11_13=0
        f_7_11=0
        f_5_8=0
        f_6_9=0
        f_9_12=0
        f_4_8=0
        f_8_12=0
        f_14=c4_list[x]
        degx = deg_mat[x]
        out_col[i][0] = degx
        f_1 = 0
        f_2 = 0
        f_3 = 0
        # x is side node
        nx1 = 0
        while nx1 < degx:
            y1 = dst_list[xnei+nx1]
            degy1 = deg_mat[y1]
            y1nei = src_dict[y1]
            tri_ey1 = edge_tri[xnei+nx1]
            ny1 = 0
            while ny1 < degy1:
                z1 = dst_list[y1nei+ny1]
                if x==z1:
                    ny1 += 1
                    continue
                tri_ez1 = edge_tri[y1nei+ny1]
                mybool = True
                if tri_ey1 > 0 and tri_ez1 > 0:
                    tmpi = 0
                    while tmpi < degx:# time complexity: O(n*d*d*d)
                        if z1 < dst_list[xnei+tmpi]:
                            break
                        if z1 == dst_list[xnei+tmpi]:
                            mybool = False # xyz is not a path, it is triangle
                        tmpi += 1
                if mybool: # xyz is a path
                    #out_col[x][1] = out_col[x][1] + 1
                    f_1 += 1
                    f_6_9 += ((degy1-1-tri_ey1-1))
                    f_9_12 += tri_ez1
                    f_4_8 += ((deg_mat[z1]-1-tri_ez1))
                    # f_8_12 += (common[z]-1);
                ny1 += 1
            nx1 += 1
        # x is middle node
        nx1 = 0
        nx2 = 0
        while nx1 < degx:
            y1 = dst_list[xnei+nx1]
            degy1 = deg_mat[y1]
            y1nei = src_dict[y1]
            tri_ey1 = edge_tri[xnei+nx1]
            ny1 = 0
            while ny1 < degy1:
                z1 = dst_list[y1nei+ny1]
                z1nei = src_dict[z1]
                tri_ez1 = edge_tri[y1nei+ny1]
                mybool = True
                if tri_ey1 > 0 and tri_ez1 > 0:
                    tmpi = 0
                    #if z1 >= dst_list[xnei+(deg_mat[x]//2)]:
                    #    tmpi += (deg_mat[x]//2)
                    while tmpi < degx:# time complexity: O(N*d*d*d)
                        if z1 < dst_list[xnei+tmpi]:
                            break
                        if z1 == dst_list[xnei+tmpi]:
                            if z1 < y1:
                                f_12_14 += (tri_ez1-1)
                                f_10_13 += ((degy1-1-tri_ez1)+(deg_mat[z1]-1-tri_ez1))
                            mybool = False # xyz is not a path, it is triangle
                        tmpi += 1
                if mybool: # if xyz is a path rather than a triangle
                    # calculate butterfly
                    nx2 = 0
                    nz1 = 0
                    degz1 = deg_mat[z1]
                    while nx2 < degx and nz1 < degz1:# time complexity: O(N*d*d*d)
                        if nx2 >= nx1:
                            break
                        y2 = dst_list[xnei+nx2]
                        z2 = dst_list[z1nei+nz1]
                        if z2 == y2:
                            f_8_12 = f_8_12 + 2
                            nx2 += 1
                        elif z2 < y2:
                            nz1 += 1
                        else:
                            nx2 += 1
                ny1 += 1
            nx2 = nx1 + 1
            while nx2 < degx:
                z1 = dst_list[xnei+nx2]
                tri_ez1 = edge_tri[xnei+nx2]
                mybool = True
                if tri_ey1 > 0 and tri_ez1 > 0:
                    tmpi = 0
                    #if z1 >= dst_list[y1nei+(deg_mat[y1]//2)]:
                    #    tmpi += (deg_mat[y1]//2)
                    while tmpi < degy1:# time complexity: O(N*d*d)
                        if z1 < dst_list[y1nei+tmpi]:
                            break
                        if z1 == dst_list[y1nei+tmpi]:
                            mybool = False # xyz is not a path, it is triangle
                        tmpi += 1
                if mybool: # if xyz is a path rather than a triangle
                    #out_col[x][2] = out_col[x][2] + 1
                    f_2 += 1
                    f_7_11 += ((degx-1-tri_ey1-1)+(degx-1-tri_ez1-1))
                    f_5_8 += ((degy1-1-tri_ey1)+(deg_mat[z1]-1-tri_ez1))
                else:
                    #out_col[x][3] = out_col[x][3] + 1
                    f_3 += 1
                    f_13_14 += ((tri_ey1-1)+(tri_ez1-1))
                    f_11_13 += ((degx-1-tri_ey1)+(degx-1-tri_ez1))
                nx2 += 1
            nx1 += 1
        # calculate f_8_12
        f_8_12 = f_8_12 - (f_2+f_3)*2
        out_col[x][14]=(f_14)
        out_col[x][13]=(f_13_14-6*f_14)/2
        out_col[x][12]=(f_12_14-3*f_14)
        out_col[x][11]=(f_11_13-f_13_14+6*f_14)/2
        out_col[x][10]=(f_10_13-f_13_14+6*f_14)
        out_col[x][9]=(f_9_12-2*f_12_14+6*f_14)/2
        out_col[x][8]=(f_8_12-2*f_12_14+6*f_14)/2
        out_col[x][7]=(f_13_14+f_7_11-f_11_13-6*f_14)/6
        out_col[x][6]=(2*f_12_14+f_6_9-f_9_12-6*f_14)/2
        out_col[x][5]=(2*f_12_14+f_5_8-f_8_12-6*f_14)
        out_col[x][4]=(2*f_12_14+f_4_8-f_8_12-6*f_14)
        out_col[x][3]=f_3
        out_col[x][2]=f_2
        out_col[x][1]=f_1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--orbit-type", help="Choose orbit type.", default="node", type=str,
                        choices=["node", "edge"],
                        required=True)
    parser.add_argument("-s", "--graphlet-size", help="Choose graphlet size.", default=4, type=int,
                        choices=[4, 5],
                        required=True)
    parser.add_argument("-i", "--input-file", help="Path of input file. Example:```-i google.txt```", default=None)
    parser.add_argument("-o", "--output-file", help="Specify the name of output file.", default=None)
    args = parser.parse_args()
    df = pd.read_csv(args.input_file, sep=" ")
    nnode = int(df.columns[0])
    nedge = int(df.columns[1])
    print("nodes: {}\nedges: {}".format(nnode, nedge))
    src = df[df.columns[0]].values.astype(int)
    dst = df[df.columns[1]].values.astype(int)
    e1 = np.concatenate([src, dst])
    e2 = np.concatenate([dst, src])
    adj_sparse = np.sparse.coo_matrix((np.full((nedge * 2,), 1.), (e1, e2))).tocsr()
    deg = adj_sparse.sum(0)[0].astype(np.int32)
    print("max degree: {}".format(deg.max()))
    tasktype = args.orbit_type
    tasknum = args.graphlet_size
    print("Counting {} orbits of graphlets on {} {}s.\n".format(tasktype, tasknum, tasktype))
    if tasktype == 'node' and tasknum == 4:
        pass
    else:
        print("There is only node orbits of graphlets on 4 nodes. "
              "Other graphlet counting will be implemented in the future.")
        import sys
        sys.exit(0)
    edf = pd.DataFrame()
    edf['src'] = src.astype(np.int32)
    edf['dst'] = dst.astype(np.int32)
    eedf = pd.DataFrame()
    eedf['src'] = np.concatenate([src, dst])
    eedf['dst'] = np.concatenate([dst, src])
    eedf = eedf.sort_values(by=['src', 'dst']).reset_index()[['src', 'dst']]
    src_dict = eedf.reset_index().groupby("src").index.min()
    src_dict = src_dict.sort_values().values
    size = len(edf['src'])
    etri = np.zeros(nedge)
    print("stage 1 - precomputing common nodes")
    start = time.time()
    edge_triangle.forall(size)(edf['src'], edf['dst'], etri, deg, eedf['dst'], src_dict)
    eedf = pd.DataFrame()
    eedf['src'] = np.concatenate([src, dst])
    eedf['dst'] = np.concatenate([dst, src])
    eedf['etri'] = np.concatenate([etri, etri])
    ndf = pd.DataFrame()
    ndf['node'] = np.array(range(nnode))
    ndf['deg'] = deg
    c_four = np.zeros(nnode).astype(np.int32)
    eedf = eedf.sort_values(by=['src', 'dst']).reset_index()[['src', 'dst', 'etri']]
    src_dict = eedf.reset_index().groupby("src").index.min()
    src_dict = src_dict.sort_values().values
    eedf['neigh'] = np.zeros(nedge * 2).astype(np.int32)
    size = len(ndf['node'])
    end = time.time()
    t1 = end-start
    print(t1)
    print("stage 2 - counting full graphlets")
    start = time.time()
    full_graphlet.forall(size)(ndf['node'], c_four, eedf['neigh'], deg, eedf['dst'], src_dict)
    ndf['c4'] = c_four
    end = time.time()
    t2 = end - start
    print(t2)
    print("stage 3 - building systems of equations")
    start = time.time()
    threadsperblock = 16
    blockspergrid = math.ceil(size / threadsperblock)
    outdf = pd.DataFrame()
    for i in range(15):
        outdf['{}'.format(i)] = np.zeros(nnode).astype(np.uint32)
    results = outdf[[str(i) for i in list(range(15))]].values
    equation_system[blockspergrid, threadsperblock](ndf['node'].values, results, ndf['deg'].values, eedf['dst'].values,
                                                    src_dict, eedf['etri'].values, ndf['c4'].values)
    #equation_system.forall(size)(ndf['node'].values, results, ndf['deg'].values, eedf['dst'].values,
    #                             src_dict, eedf['etri'].values, ndf['c4'].values)
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, header=None, index=False, sep=" ")
    end = time.time()
    t3 = end-start
    print(t3)
    print("total: {}".format(t1+t2+t3))
