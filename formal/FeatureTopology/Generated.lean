import FeatureTopology.Graph

set_option maxHeartbeats 10000000
set_option maxRecDepth 4096

/- Certificate SHA256: 7f728d2af7363f1dd203ba293e9cbf0bd41419eba2092907014b213e1393ce3f.
   This checks the abstract supplied graph, not Python's geometric construction. -/
namespace FeatureTopology.Generated

def certificateGraph : FeatureTopology.FiniteGraph :=
  ⟨38, [(0, 1), (0, 8), (1, 3), (2, 4), (2, 7), (3, 5), (4, 6), (5, 6), (7, 12), (8, 9), (9, 10), (10, 11), (11, 13), (12, 14), (13, 18), (14, 15), (15, 16), (16, 17), (17, 19), (18, 25), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 28), (25, 26), (26, 27), (27, 29), (28, 33), (29, 30), (30, 31), (31, 32), (32, 34), (33, 37), (34, 35), (35, 36), (36, 37)]⟩

theorem certificate_graph_checked :
    FeatureTopology.checkGraph certificateGraph 1 1 = true := by decide

end FeatureTopology.Generated
