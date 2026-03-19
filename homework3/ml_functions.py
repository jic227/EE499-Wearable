import math
import random


def euclidean_distance(point1, point2):
    """
    Compute Euclidean distance between two points.
    """
    total = 0
    for i in range(len(point1)):
        total += (point1[i] - point2[i]) ** 2
    return math.sqrt(total)


def mean_point(points):
    """
    Compute the center of a cluster by averaging each dimension.
    """
    dims = len(points[0])
    center = []

    for d in range(dims):
        dim_sum = 0
        for p in points:
            dim_sum += p[d]
        center.append(dim_sum / len(points))

    return center


def kmeans(data, k, max_iters=100):
    """
    Run k-means clustering on a dataset.
    Returns:
        centroids
        labels
    """
    # pick k random starting centroids
    centroids = random.sample(data, k)

    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        labels = []

        # assign each point to closest centroid
        for point in data:
            distances = []
            for centroid in centroids:
                distances.append(euclidean_distance(point, centroid))

            closest_index = distances.index(min(distances))
            clusters[closest_index].append(point)
            labels.append(closest_index)

        new_centroids = []

        # recompute each centroid
        for i in range(k):
            if len(clusters[i]) == 0:
                new_centroids.append(centroids[i])
            else:
                new_centroids.append(mean_point(clusters[i]))

        # stop if centroids do not change
        if new_centroids == centroids:
            break

        centroids = new_centroids

    return centroids, labels

def knn(train_data, train_labels, test_point, k):
    """
    K-Nearest Neighbors classifier.
    Returns predicted label for test_point.
    """
    distances = []

    # compute distance from test point to each training point
    for i in range(len(train_data)):
        dist = euclidean_distance(train_data[i], test_point)
        distances.append((dist, train_labels[i]))

    # sort by smallest distance
    distances.sort(key=lambda x: x[0])

    # get the k closest labels
    nearest_labels = []
    for i in range(k):
        nearest_labels.append(distances[i][1])

    # majority vote
    return max(set(nearest_labels), key=nearest_labels.count)

def segment_mean(data, start, end):
    """
    Compute mean of a segment of the data from start to end.
    """
    segment = data[start:end]
    return sum(segment) / len(segment)


def segment_error(data, start, end):
    """
    Compute sum of squared error for one segment.
    """
    mean = segment_mean(data, start, end)
    error = 0

    for i in range(start, end):
        error += (data[i] - mean) ** 2

    return error


def find_best_split(data, start, end):
    """
    Find the best split point in one segment by minimizing total error.
    """
    best_index = None
    best_error = float("inf")

    for split in range(start + 1, end):
        left_error = segment_error(data, start, split)
        right_error = segment_error(data, split, end)
        total_error = left_error + right_error

        if total_error < best_error:
            best_error = total_error
            best_index = split

    return best_index, best_error


def cpa(data, max_changes=1):
    """
    Simple change point analysis.
    Repeatedly finds the best split point in the largest current segment.
    Returns list of change point indices.
    """
    segments = [(0, len(data))]
    change_points = []

    for _ in range(max_changes):
        best_segment = None
        best_split = None
        best_error = float("inf")

        for start, end in segments:
            if end - start > 2:
                split, error = find_best_split(data, start, end)

                if split is not None and error < best_error:
                    best_error = error
                    best_split = split
                    best_segment = (start, end)

        if best_split is None:
            break

        change_points.append(best_split)

        segments.remove(best_segment)
        segments.append((best_segment[0], best_split))
        segments.append((best_split, best_segment[1]))

    change_points.sort()
    return change_points   