'''
AUTOR: Paulina Kimak
NR INDEKSU: 292511
Treść zadania:
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
def kmp_search_with_stop(text, pattern, comm, nr_procesu):
    if pattern == "":
        return 1

    pi = prefix_function(pattern)
    j = 0

    for i in range(len(text)):

        # co pewien czas sprawdzamy, czy inny proces już znalazł wynik
        if i % 1000 == 0:
            if comm.iprobe(source=MPI.ANY_SOURCE, tag=TAG_STOP):
                wynik, kto_znalazl = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_STOP)
                print(f"Proces {nr_procesu}: kończę, bo proces {kto_znalazl} znalazł klucz", flush=True)
                return -1

        while j > 0 and text[i] != pattern[j]:
            j = pi[j - 1]

        if text[i] == pattern[j]:
            j += 1

        if j == len(pattern):
            return 1

    return 0

def podziel_tekst(text, liczba_procesow, dlugosc_wzorca):
    n = len(text)
    overlap = max(0, dlugosc_wzorca - 1)
    rozmiar = n // liczba_procesow

    fragmenty = []

    for nr in range(liczba_procesow):
        start = nr * rozmiar
        if nr == liczba_procesow - 1:
            end = n
        else:
            end = (nr + 1) * rozmiar
        # zakładka - żeby znaleźć wzorzec na granicy fragmentów
        end = min(n, end + overlap)
        fragment = text[start:end]
        fragmenty.append((start, end, fragment))

    return fragmenty

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
klucz = comm.bcast(klucz, root=0)

badany_przedzial = comm.scatter(lista_przedzialow, root=0)

start, end, fragment = badany_przedzial

znaleziono = False
wynik = None

# Szukanie współbieżne w przydzielonym fragmencie
wynik_wyszukiwania = kmp_search_with_stop(fragment, klucz, comm, nr_procesu)

if wynik_wyszukiwania == 1:
    znaleziono = True
    wynik = (1, nr_procesu)

    # rozgłoszenie do pozostałych procesów, że znaleziono słowo
    for p in range(liczba_procesow):
        if p != nr_procesu:
            comm.send((1, nr_procesu), dest=p, tag=TAG_STOP)

    print(f"Proces {nr_procesu}: znalazłem klucz w przedziale {start}:{end}", flush=True)

elif wynik_wyszukiwania == -1:
    wynik = (1, None)

else:
    wynik = (0, None)

# Zebranie wyniku
wyniki = comm.gather(wynik, root=0)

if nr_procesu == 0:
    czy_znaleziono = False

    for wynik_procesu in wyniki:
        if wynik_procesu[0] == 1:
            czy_znaleziono = True

    if czy_znaleziono:
        print(1)
    else:
        print(0)