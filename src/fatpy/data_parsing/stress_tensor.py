import numpy as np

class stress_tensor:
    def __init__(self, sx, sy, sz, sxy, syz, sxz):
        #constructor
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.sxy = sxy
        self.syz = syz
        self.sxz = sxz

        # Compute eigenvalues
        self.stress_tensor = np.array([
            [self.sx, self.sxy, self.sxz],
            [self.sxy, self.sy, self.syz],
            [self.sxz, self.syz, self.sz]
        ])
        self.eigenvalues = np.linalg.eigvals(self.stress_tensor)

    def s1(self):
        """
        Calculate 1st principal stress (maximum eigenvalue).
        """
        return np.max(self.eigenvalues)

    def s2(self):
        """
        Calculate 2nd principal stress (middle eigenvalue).
        """
        return np.partition(self.eigenvalues, 1)[1]

    def s3(self):
        """
        Calculate 3rd principal stress (minimum eigenvalue).
        """
        return np.min(self.eigenvalues)

    def von_mises(self):
        """
        Calculate von-mises stress.
        """
        vm = np.sqrt(
            0.5 * (
                (self.sx - self.sy)**2 +
                (self.sy - self.sz)**2 +
                (self.sz - self.sx)**2 +
                6 * (self.sxy**2 + self.syz**2 + self.sxz**2)
            )
        )
        return vm

    def tresca(self):
        """
        Calculate tresca stress.
        """
        s1_val = self.s1()
        s2_val = self.s2()
        s3_val = self.s3()
        
        tresca_stress = max(abs(s1_val - s2_val), abs(s2_val - s3_val), abs(s3_val - s1_val))
        return tresca_stress

    def max_principal_stress(self):
        """
        Calculate maximum principal stress.
        """
        return max(self.s1(), self.s2(), self.s3())
    
    def hydrostatic_mean_stress(self):
        """
        Calculate hydrostatic mean stress.
        """
        return (self.s1()+self.s2()+self.s3())
    def sign(self):
        """
        Calculate sign of the hydrostatic mean stress.
        1-e5 for zero!
        """
        return np.sign(self.hydrostatic_mean_stress()+1e-5)
    def signed_von_mises(self):
        """
        Calculate signed von mises
        """
        value=self.sign()*self.von_mises()
        return value