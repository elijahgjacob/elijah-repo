package enigma;

import static enigma.EnigmaException.*;

/** Represents a permutation of a range of integers starting at 0 corresponding
 *  to the characters of an alphabet.
 *  @author P.N.Hilfinger
 */
class Permutation {

    /** Set this Permutation to that specified by CYCLES, a string in the
     *  form "(cccc) (cc) ..." where the c's are characters in ALPHABET, which
     *  is interpreted as a permutation in cycle notation.  Characters in the
     *  alphabet that are not included in any cycle map to themselves.
     *  Whitespace is ignored. */
    Permutation(String cycles, Alphabet alphabet) {
        _alphabet = alphabet;
        _cycles = cycles;
    }

    /** Add the cycle c0->c1->...->cm->c0 to the permutation, where CYCLE is
     *  c0c1...cm. */
    private void addCycle(String cycle) {
        _cycles += " " + cycle;
    }

    /** Return the value of P modulo the size of this permutation. */
    final int wrap(int p) {
        int r = p % size();
        if (r < 0) {
            r += size();
        }
        return r;
    }

    /** Returns the size of the alphabet I permute. */
    int size() {
        return _alphabet.size();
    }

    /** Return the result of applying this permutation to P modulo the
     *  alphabet size. */
    int permute(int p) {
        if (_cycles.indexOf(p+1) == -1) {
            return p;
        }
        if (_cycles.charAt(_cycles.indexOf(p) + 1) == (')')) {
            int index = _cycles.indexOf("(");
            int lastInd = -1;
            while (index < _cycles.indexOf(p) && index != -1) {
                lastInd = index;
                index = _cycles.indexOf("(", index + 1);
            }
            return _cycles.charAt(lastInd + 1);
        } else {
            return _cycles.charAt(_cycles.indexOf(p) + 1);
        }
    }

    /** Return the result of applying the inverse of this permutation
     *  to  C modulo the alphabet size. */
    int invert(int c) {
        if (_cycles.indexOf(c)==-1) {
            return c;
        }
        if (_cycles.charAt(_cycles.indexOf(c) - 1) == ('(')) {
            int index = _cycles.indexOf(")");
            int lastInd = -1;
            while (index < _cycles.indexOf(c)) {
                lastInd = index;
                index = _cycles.indexOf(")", index + 1);
            }
            return _cycles.charAt(index - 1);
        } else {
            return _cycles.charAt(_cycles.indexOf(c) - 1);
        }
    }

    /** Return the result of applying this permutation to the index of P
     *  in ALPHABET, and converting the result to a character of ALPHABET. */
    char permute(char p) {
        int x = _alphabet.toInt(p);
        x = permute(x);
        return _alphabet.toChar(x);
    }

    /** Return the result of applying the inverse of this permutation to C. */
    char invert(char c) {
        int y = alphabet().toInt(c);
        y = invert(y);
        return alphabet().toChar(y);
    }

    /** Return the alphabet used to initialize this Permutation. */
    Alphabet alphabet() {
        return _alphabet;
    }

    /** Return true iff this permutation is a derangement (i.e., a
     *  permutation for which no value maps to itself). */
    boolean derangement() {
        for (int i = 0; i < alphabet().size(); i++) {
            char c = alphabet().toChar(i);
            if (c == permute(c)){
                return false;
            }
        }
        return true;
    }

    /** Alphabet of this permutation. */
    private Alphabet _alphabet;

    private String _cycles;
}
