'''
W zadaniu chodzi o napisanie pewnej wersji przeszukiwania
pliku pod kątem słowa kluczowego. Na wejściu powinniśmy dostać:
a. ścieżkę pliku;
b. słowo kluczowe;
potem dzielimy plik na części (zachodzące na siebie
jak ,,łuski'' - aby poszukiwane słowo nie ,,ukryło'' 
się ,,na styku'' podzielonych plików).
  potem zasadniczy algorytm wyszukiwania:
  możemy skorzystać z:
    algorytmu:
      Boyera-Moora;
      Knutha-Morrisa-Pratta;
a potem ,,zbieramy'' wyniki.
'''
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
nr_procesu = comm.Get_rank()
liczba_procesow = comm.Get_size()

TAG_STOP = 99 # to jest tag broadcastu


def prefix_function(pattern):
    pi = [0] * len(pattern)
    j = 0

    for i in range(1, len(pattern)):
        while j > 0 and pattern[i] != pattern[j]:
            j = pi[j - 1]

        if pattern[i] == pattern[j]:
            j += 1

        pi[i] = j

    return pi

# Knuth-Morris-Pratt algorytm
def kmp_search(text, pattern):
    if pattern == "":
        return True

    pi = prefix_function(pattern)
    j = 0

    for i in range(len(text)):
        while j > 0 and text[i] != pattern[j]:
            j = pi[j - 1]

        if text[i] == pattern[j]:
            j += 1

        if j == len(pattern):
            return True

    return False

def podziel_tekst(text, liczba_procesow, dlugosc_wzorca):
    n = len(text)
    overlap = max(0, dlugosc_wzorca - 1)
    rozmiar = n // liczba_procesow

    przedzialy = []

    for nr in range(liczba_procesow):
        start = nr * rozmiar
        if nr == liczba_procesow - 1:
            end = n
        else:
            end = (nr + 1) * rozmiar
        # zakładka - żeby znaleźć wzorzec na granicy fragmentów
        end = min(n, end + overlap)
        przedzialy.append((start, end))

    return przedzialy

# Proces 0 czyta dane 
if nr_procesu == 0:
    if len(sys.argv) != 3:
        print("Użycie: mpirun -np <liczba_procesow> python3 program.py <plik> <klucz>")
        sys.exit(1)

    nazwa_pliku = sys.argv[1]
    klucz = sys.argv[2]

    with open(nazwa_pliku, "r", encoding="utf-8") as f:
        tekst = f.read()
    lista_przedzialow = podziel_tekst(tekst, liczba_procesow, len(klucz))
else:
    tekst = None
    klucz = None
    lista_przedzialow = None

# Rozsyłanie danych
tekst = comm.bcast(tekst, root=0)
klucz = comm.bcast(klucz, root=0)

badany_przedzial = comm.scatter(lista_przedzialow, root=0)

start, end = badany_przedzial
fragment = tekst[start:end]

znaleziono = False
wynik = None

# Szukanie współbieżne w przydzielonym fragmencie
while True:
    # sprawdzenie, czy inny proces już znalazł wynik
    if comm.iprobe(source=MPI.ANY_SOURCE, tag=TAG_STOP):
        wynik = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_STOP)
        break
    
    # Knuth-Morris-Pratt algorytm
    if kmp_search(fragment, klucz):
        znaleziono = True
        wynik = 1
        
        # rozgłoszenie do pozostałych procesów, że znaleziono słowo
        for p in range(liczba_procesow):
            if p != nr_procesu:
                comm.send(wynik, dest=p, tag=TAG_STOP)
        print(f"Proces {nr_procesu}: znalazłem klucz w przedziale {start}:{end}")
        break
    else:
        wynik = 0
        break

# Zebranie wyniku
wyniki = comm.gather(wynik, root=0)

if nr_procesu == 0:
    if 1 in wyniki:
        print(1)
    else:
        print(0)