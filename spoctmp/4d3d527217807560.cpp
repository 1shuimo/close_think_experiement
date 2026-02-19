#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <set>
#include <map>
#include <queue>
#include <stack>
#include <list>
#include <fstream>
#include <climits>
#include <cassert>
#include <iomanip>
#include <sstream>
#include <bitset>
using namespace std;

string tostr(int n) {
    stringstream rr;
    rr << n;
    return rr.str();
}

int check_lucky(long long n) {
    while (n > 0) {
        if (n % 10 == 8) return 1;
        n = n / 10;
    }
    return 0;
}

int main() {
    long long a;
    cin >> a;
    int res = 0;
    while (!check_lucky(abs(a)) || res == 0) {
        res++;
        a++;
    }
    cout << res << endl;
    return 0;
}
