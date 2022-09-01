# Copyright (c) OpenXRLab. All rights reserved.
class QuickUnion(object):
    """QuickUnion."""
    def __init__(self, vtx):
        assert vtx is not None
        self.vtx = vtx
        self.tree = [-1] * len(self.vtx)
        self.keys = dict(zip(vtx, [i for i in range(len(vtx))]))

    def add_edge(self, edge):
        v1, v2 = edge[0], edge[1]
        fv1, fv2 = self.root(v1), self.root(v2)
        if fv1 != fv2:
            self.tree[fv2] = fv1

    def clusters(self):
        kcls = {}
        for v in self.vtx:
            root = self.root(v)
            if root not in kcls:
                kcls[root] = []
            kcls[root].append(v)
        cls = []
        for key in kcls:
            cls.append(kcls[key])
        return cls

    def root(self, v):
        assert v in self.vtx
        key = self.keys[v]
        while self.tree[key] != -1:
            key = self.tree[key]
        return key


def covisible_clustering(image_ids, sfm):
    """Covisible Clustering
    Args:
        image_ids (list[int]): Retrieved image ids
        sfm (Reconstruction): Sparse SfM model
    Returns:
        List: Multi covisible images list
    """
    qun = QuickUnion(image_ids)
    idset = set(image_ids)
    for id in image_ids:
        covidset = set(sfm.covisible_images(id, 0))
        covidset &= idset
        for cid in covidset:
            qun.add_edge([id, cid])
    clusters = qun.clusters()
    clusters = sorted(clusters, key=len, reverse=True)
    # Sort each cluster according to the order of retrieval
    keys = dict(zip(image_ids, [i for i in range(len(image_ids))]))
    clusters = [sorted(cluster, key=lambda k: keys[k]) for cluster in clusters]
    return clusters


def scene_retrieval(image_ids, map, size):
    """Covisible Clustering
    Args:
        image_ids (list[int]): Retrieved image ids
        map (Reconstruction): Sparse SfM model
        size (int): Scene size
    Returns:
        List: Multi covisible scene
    """
    scenes = []
    processed = dict(zip(image_ids, [False] * len(image_ids)))
    image_id_set = set(image_ids)
    for id in image_ids:
        if processed[id]: continue
        covids = map.covisible_images(id)

        retrieved_image_ids = [id for id in covids if id in image_id_set]
        for id in retrieved_image_ids:
            processed[id] = True
        covisible_image_ids = [id for id in covids if id not in image_id_set]
        if len(retrieved_image_ids) < size:
            retrieved_image_ids += covisible_image_ids[:size -
                                                       len(retrieved_image_ids
                                                           )]
        scenes.append(retrieved_image_ids)
    return scenes
