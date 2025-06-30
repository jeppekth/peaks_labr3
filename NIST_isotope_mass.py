class NIST_isotope_mass:
    def __init__(self, filename):
        self._isotope_dict = {}
        with open(filename) as f:
            at_data = False

            symbol = ""

            relative_atomic_mass_to_keV: float = 1.11779292e7 / 12.

            for l in f:
                if l.isspace() or l == "":
                    continue

                if not at_data and "_" not in l:
                    continue
                elif not at_data and "_" in l:
                    at_data = True
                    continue
                elif at_data and "_" in l:
                    continue

                if not l[0:3].isspace():
                    symbol = l[4:6].replace(" ", "")
                    self._isotope_dict[symbol] = {"Z": int(l[0:3]), "isotopes mass": {}}
                
                A = int(l[8:11])
                mass_string_end = l.find("(")
                mass = float(l[13:mass_string_end])
                mass = mass * relative_atomic_mass_to_keV

                self._isotope_dict[symbol]["isotopes mass"][A] = mass
    
    def get_isotope_mass(self, symbol: str, A: int) -> float:
        return self._isotope_dict[symbol]["isotopes mass"][A]
    
    def get_Z(self, symbol: str) -> int:
        return self._isotope_dict[symbol]["Z"]





