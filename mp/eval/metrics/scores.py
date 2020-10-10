# ------------------------------------------------------------------------------
# Definition of a score metrics for classification and segmentation, taking
# in tp, tn, fn and fp as inputs. For segmentation, these refer to pixel/voxel
# values for one example.
# ------------------------------------------------------------------------------

class ScoreAbstract:
    def __init__(self):
        self.name = self.__class__.__name__

    def eval(self, tp, tn, fn, fp):
        raise NotImplementedError

class ScoreDice(ScoreAbstract):
    def eval(self, tp, tn, fn, fp):
        if tp == 0:
            if fn+fp > 0:
                return 0.
            else:
                return 1.
        return (2*tp)/(2*tp+fp+fn)

class ScoreIoU(ScoreAbstract):
    def eval(self, tp, tn, fn, fp):
        if tp == 0:
            if fn+fp > 0:
                return 0.
            else:
                return 1.
        return tp/(tp+fp+fn)

class ScorePrecision(ScoreAbstract):
    def eval(self, tp, tn, fn, fp):
        if tp == 0:
            if fp > 0:
                return 0.
            else:
                return 1.
        return tp/(tp+fp)

class ScorePPV(ScorePrecision):
    pass

class ScoreRecall(ScoreAbstract):
    def eval(self, tp, tn, fn, fp):
        if tp == 0:
            if fp > 0:
                return 0.
            else:
                return 1.
        return tp/(tp+fn)

class ScoreSensitivity(ScoreRecall):
    pass

class ScoreTPR(ScoreRecall):
    pass

class ScoreSpecificity(ScoreAbstract):
    def eval(self, tp, tn, fn, fp):
        if tn == 0:
            if fp > 0:
                return 0.
            else:
                return 1.
        return tn/(tn+fp)

class ScoreTNR(ScoreSpecificity):
    pass