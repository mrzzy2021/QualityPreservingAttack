from model.frhead import tools

class NTUGraph:
    def __init__(self, labeling_mode='spatial'):
        num_node = 25
        self_link = [(i, i) for i in range(num_node)]
        inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                            (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                            (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                            (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
        inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
        outward = [(j, i) for (i, j) in inward]
        neighbor = inward + outward
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A


class SMPLGraph:
    def __init__(self, labeling_mode='spatial'):
        num_node = 22
        self_link = [(i, i) for i in range(num_node)]
        # index starting from 1 to 22
        inward_ori_index = [(1, 2), (2, 5), (5, 8), (8, 11),
                            (1, 3), (3, 6), (6, 9), (9, 12),
                            (1, 4), (4, 7), (7, 10), (10, 13), (13, 16),
                            (10, 14), (14, 17), (17, 19), (19, 21),
                            (10, 15), (15, 18), (18, 20), (20, 22)]
        inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
        outward = [(j, i) for (i, j) in inward]
        neighbor = inward + outward
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A


class HDM05Graph:
    def __init__(self, labeling_mode='spatial'):
        num_node = 25
        self_link = [(i, i) for i in range(num_node)]
        # index starting from 1 to 25
        inward_ori_index = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 6),
                            (1, 7), (7, 8), (8, 9), (9, 10), (1, 11),
                            (1, 12), (12, 13), (13, 14), (14, 15),
                            (14, 16), (16, 17), (17, 18), (18, 19), (19, 20),
                            (14, 21), (21, 22), (22, 23), (23, 24), (24, 25),]
        inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
        outward = [(j, i) for (i, j) in inward]
        neighbor = inward + outward
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A


class STYLEORIGraph:
    def __init__(self, labeling_mode='spatial'):
        num_node = 23
        self_link = [(i, i) for i in range(num_node)]
        # index starting from 1 to 23
        inward_ori_index = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
                            (5, 8), (8, 9), (9, 10), (10, 11),
                            (5, 12), (12, 13), (13, 14), (14, 15),
                            (1, 16), (16, 17), (17, 18), (18, 19),
                            (1, 20), (20, 21), (21, 22), (22, 23)]
        inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
        outward = [(j, i) for (i, j) in inward]
        neighbor = inward + outward
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A