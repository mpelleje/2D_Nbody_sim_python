#include "unfolding.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

using std::vector;
using std::list;
using std::set;
using std::pair;

#include <chrono>
#include <string>

class Timer {
private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = std::chrono::duration<double>;

    std::unordered_map<std::string, Duration> measurements;
    TimePoint lastResetTime;
    
    bool active;

public:
    Timer(bool set_active=true)
    {
        active = set_active;
        this->resetTimer();
    }
    
    void resetTimer() {
        lastResetTime = Clock::now();
    }

    void measureDT(const std::string& name) {
        if(!active)
            return;
        
        auto currentTime = Clock::now();
        auto duration = std::chrono::duration_cast<Duration>(currentTime - lastResetTime);
        
        if(measurements.count(name) == 0)
            measurements[name] = Duration(0);
        measurements[name] += duration;
        lastResetTime = currentTime;
    }

    double getTotalTime(const std::string& name) const {
        auto iter = measurements.find(name);
        if (iter != measurements.end()) {
            return iter->second.count();
        }
        return 0.0; // Return 0 if the name is not found
    }
    
    void printResults() const {
        Duration total_time(0);
        
        for (const auto& measurement : measurements) {
            total_time += measurement.second;
        }
        
        printf("%15s %10.3f seconds (%5.1f\%)\n", "Total", total_time.count(), 100.);
        for (const auto& measurement : measurements) {
            double percent = measurement.second.count()*100./total_time.count();
            printf("%15s %10.3f seconds (%5.1f\%)\n", measurement.first.c_str(), measurement.second.count(), percent);
        }
    }
};

void myassert(bool condition, const char * message)
{
    if(!condition) {
        printf(message);
        fflush(stdout);
        assert(0);
    }
}


template <typename TFloat>
struct SimpStruct {
    int64_t id;
    TFloat area;
};

template <typename TFloat>
struct SimpCompare {
    bool operator()(const SimpStruct<TFloat>& lhs, const SimpStruct<TFloat>& rhs) const {
        return lhs.area > rhs.area;
    }
};


template<typename T>
void eraseElement(std::vector<T>& vec, typename std::vector<T>::size_type index) {
    if (index < vec.size() - 1) {
        vec[index] = std::move(vec.back()); // Replace element with the last element
    }
    vec.pop_back(); // Remove the last element
}

template <typename TFloat>
inline void wrapped_dx(TFloat *x1, TFloat *x2, TFloat L, int ndim, TFloat *dxout)
{
    for(int i=0; i<ndim; i++) {
        TFloat dx = x2[i] - x1[i];
        if(dx >= L/2.)
            dx -= L;
        else if(dx < -L/2.)
            dx += L;
        dxout[i] = dx;
    }
}

template <typename TFloat>
inline TFloat wrapped_dr2(TFloat *x1, TFloat *x2, TFloat L, int ndim)
{
    TFloat dx[3];
    wrapped_dx(x1, x2, L, ndim, dx);
    
    TFloat dr2 = 0.;
    for(int i=0; i<ndim; i++) {
        dr2 += dx[i]*dx[i];
    }
    return dr2;
}

template <typename TFloat>
inline TFloat cross_product_2d(TFloat *a, TFloat *b)
{
    return a[0] * b[1]  - a[1] * b[0];
}

template <typename TFloat>
inline TFloat triangle_parity(TFloat *x1, TFloat *x2, TFloat *x3, TFloat L)
{
    TFloat a[2], b[2];
    
    wrapped_dx(x1, x2, L, 2, a);
    wrapped_dx(x1, x3, L, 2, b);
    
    return cross_product_2d(a, b);
}


template <typename TFloat>
inline void cross_product_3d(TFloat *a, TFloat *b, TFloat *out)
{
    out[0] = a[1] * b[2]  - a[2] * b[1];
    out[1] = a[2] * b[0]  - a[0] * b[2];
    out[2] = a[0] * b[1]  - a[1] * b[0];
}    

template <typename TFloat>
inline TFloat tet_parity(TFloat *x1, TFloat *x2, TFloat *x3, TFloat *x4, TFloat L)
{
    TFloat a[3], b[3], c[3], normal[3];
    
    wrapped_dx(x1, x2, L, 3, a);
    wrapped_dx(x1, x3, L, 3, b);
    wrapped_dx(x1, x4, L, 3, c);
        
    cross_product_3d(a, b, normal);
    
    TFloat parity = normal[0] * c[0] + normal[1] * c[1] + normal[2] * c[2];

    return parity;
}


template <typename TFloat>
inline TFloat norm2_2d(TFloat *dx)
{
    return dx[0]*dx[0] + dx[1]*dx[1];
}

template <typename TFloat>
inline int64_t triangle_shortest_side(TFloat *x1, TFloat *x2, TFloat *x3, TFloat L)
{
    TFloat l2[3];
    
    l2[0] = wrapped_dr2(x2, x3, L, 2);
    l2[1] = wrapped_dr2(x1, x3, L, 2);
    l2[2] = wrapped_dr2(x1, x2, L, 2);
    
    int64_t imin = 0;
    for(int64_t i=1; i<3; i++){
        if(l2[i] < l2[imin])
            imin = i;
    }
    
    return imin;
}

template <typename TFloat>
inline pair<int64_t, int64_t> tet_shortest_side(TFloat *x1, TFloat *x2, TFloat *x3, TFloat *x4, TFloat L)
{
    TFloat l2min = 1e38; // this is almost max float32
    int64_t iA=0, iB=0;
    
    TFloat* x[4] = {x1, x2, x3, x4};
    
    for(int i=0; i<3; i++)
        for(int j=i+1; j<4; j++)
        {
            TFloat l2 = wrapped_dr2(x[i], x[j], L, 3);
            if(l2 < l2min) {
                l2min = l2;
                iA = i;
                iB = j;
            }
            
            //printf("side-check: %d %d, %.5f, (%d, %d, %.5f)\n", i, j, l2, iA, iB, l2min);
            //printf("xA: %.2f, %.2f, %.2f xB %.2f, %.2f, %.2f\n", x[i][0], x[i][1], x[i][2], x[j][0], x[j][1], x[j][2]);
            //fflush(stdout);
        }
    
    return pair<int64_t, int64_t> {iA,iB};
}

inline int64_t find_in(int64_t* a, int64_t nelements, int64_t number)
{
    for(int64_t i=0; i<nelements; i++){
        if(a[i] == number)
            return i;
    }
    return -1;
}

void merge_trilists(int64_t ikeep, int64_t imerge, list<int64_t> *trilist_keep, list<int64_t> *trilist_merge, int64_t *tri, list<int64_t> *defects)
{
    
    set<int64_t> third_vert;
    
    for (auto it = trilist_keep->begin(); it != trilist_keep->end();) {
        int64_t *ivert = &tri[3*(*it)]; 
        
        // Remove degenerate triangles from keep list
        // Also Check for topological defects
        // If the same triangle appears twice, we have a defect
        // and also need to do an additional merge
        // This can happen when things got merged in a loop like manner
        if(find_in(ivert, 3, imerge) >= 0) {
            // figure out the third vertex
            for(int i=0; i<3; i++) {
                if((ivert[i] != ikeep) && (ivert[i] != imerge)){
                    // the triangle has vertices (ikeep, imerge, ivert[i])
                    if(third_vert.count(ivert[i]) >= 1)
                        defects->push_back(ivert[i]);
                    
                    third_vert.insert(ivert[i]);
                }
            }
            
            it = trilist_keep->erase(it);
        }
        else {
            it++;
        }
    }

    for (auto it = trilist_merge->begin(); it != trilist_merge->end();) {
        int64_t *ivert = &tri[3*(*it)]; 
        
        /* Remove degenerate triangles */
        if(find_in(ivert, 3, ikeep)  >= 0) {
            ivert[0] = ivert[1] = ivert[2] = -1;
        }
        
        else {
            // replace imerge by ikeep in non-degenerate triangles 
            for(int i=0; i<3; i++)
            {
                if(ivert[i] == imerge)
                    ivert[i] = ikeep;
            }
            
            // Move the triangle towards the trilist of the kept vertex
            trilist_keep->push_back(*it);
        }
        it = trilist_merge->erase(it);
    }
}

struct pair_hash {
    template <class T1, class T2>
    size_t operator() (const pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};

void merge_tetlists(int64_t ikeep, int64_t imerge, list<int64_t> *tetlist_keep, list<int64_t> *tetlist_merge, int64_t *tets, list< pair<int64_t,int64_t> > *defects, Timer* timer = nullptr)
{
    list<pair<int64_t, int64_t> > degenerate_tets;
        
    for (auto it = tetlist_keep->begin(); it != tetlist_keep->end();) {
        int64_t *ivert = &tets[4*(*it)]; 
        
        if(find_in(ivert, 4, imerge) >= 0) {
            // have a post-merge degenerate tet here, remember the two other vertices
           
            int64_t iA=-1, iB=-1;
            for(int i=0; i<4; i++) {
                if((ivert[i] != ikeep) && (ivert[i] != imerge)) {
                    if(iA == -1)
                        iA = ivert[i];
                    else
                        iB = ivert[i];
                }
            }
            
            myassert((iA >= 0) && (iB >= 0), "Something went wrong about identifying the degenerate tets\n");
            degenerate_tets.push_back(pair<int64_t, int64_t>{MIN(iA, iB), MAX(iA, iB)});
            
            it = tetlist_keep->erase(it);
        }
        else {
            it++;
        }
    }
    
    if(timer != nullptr)
        timer->measureDT("LoopKeeplist");
    
    // This scheme for detecting defects does not yet work in 3D!
    
    // Detect whether any degenerate tet is a duplicate. If so, removing it
    // will create a defect and we have to remember that
    
    /*
    for (auto it = degenerate_tets.begin(); it != degenerate_tets.end(); ++it) {
        auto a12 = *it;

        for (auto next_it = std::next(it); next_it != degenerate_tets.end(); ++next_it) {
            auto b12 = *next_it;
            
            //printf(" %d %d (%d %d) \n", ikeep, imerge, a12, b12);
            
            if((a12.first == b12.first) && (a12.second == b12.second)) {
                defects->push_back(a12.first);
                //defects->push_back(a12.second);
                printf("Got a defect! %d %d %d %d\n", ikeep, imerge, a12.first, a12.second);
                ndefect++;
                fflush(stdout);
            }
        }
    }*/
    
    std::unordered_map<pair<int, int>, int, pair_hash> pairCount;

    // Count the occurrences of each pair
    for (auto& pair : degenerate_tets) {
        pairCount[pair]++;
    }
    
    /*printf("pairs: ");
    for (const auto& pair : pairCount) {
        printf("%d ", pair.second);
    }
    printf("\n");
    fflush(stdout);*/
    
    for (auto& pair_and_count : pairCount) {
        if(pair_and_count.second >= 3) {
            defects->push_back(pair_and_count.first);
        }
    }
    
    if(timer != nullptr)
        timer->measureDT("LoopDefects");
    
    
    //printf("ndefect: %d\n", ndefect);
    //fflush(stdout);
    

    for (auto it = tetlist_merge->begin(); it != tetlist_merge->end();) {
        int64_t *ivert = &tets[4*(*it)]; 
        
        /* Remove degenerate tets */
        if(find_in(ivert, 4, ikeep)  >= 0) {
            ivert[0] = ivert[1] = ivert[2] = ivert[3] = -1;
        }
        else {
            // replace imerge by ikeep in non-degenerate tets 
            for(int i=0; i<4; i++)
            {
                if(ivert[i] == imerge)
                    ivert[i] = ikeep;
            }
            
            // Move the triangle towards the trilist of the kept vertex
            tetlist_keep->push_back(*it);
        }
        it = tetlist_merge->erase(it);
    }
    
    if(timer != nullptr)
        timer->measureDT("LoopMergeList");
}


template <typename TFloat>
void unfold2d(TFloat L, int npart, int ntri, TFloat* pos, TFloat* mass, int64_t* idptr, int64_t* tri)
{
    vector<list<int64_t>> part_to_tri;
    
    part_to_tri.resize(npart);

    for (int64_t itri=0; itri<ntri; itri++)
    {
        if(tri[3*itri] == -1) // deactivated triangle
            continue; 
        for(int ivert=0; ivert<3; ivert++) {
            part_to_tri[tri[3*itri+ivert]].push_back(itri);
        }
    }
    
    list<int64_t> tris_to_check;
    for (int64_t itri=0; itri<ntri; itri++)
    {
        int64_t *ivert = &tri[3*itri];
        if(ivert[0] == -1) // deactivated triangle
            continue; 
        
        TFloat parity = triangle_parity(&pos[2*ivert[0]], &pos[2*ivert[1]], &pos[2*ivert[2]], L);
        if(parity < 0.)
            tris_to_check.push_back(itri);
    }
    
    while (! tris_to_check.empty())
    {
        int64_t itri = tris_to_check.front();
        tris_to_check.pop_front();
        
        int64_t *ivert = &tri[3*itri];
        
        if(ivert[0] == -1) // deactivated triangle
            continue; 
        
        TFloat parity = triangle_parity(&pos[2*ivert[0]], &pos[2*ivert[1]], &pos[2*ivert[2]], L);

        if(parity < 0.)
        {
            list< pair<int64_t, int64_t> > tomerge;
            
            
            // We keep the vertex opposite of the shortest side
            int64_t imin = triangle_shortest_side(&pos[2*ivert[0]], &pos[2*ivert[1]], &pos[2*ivert[2]], L);
            
            // We merge the two vertices that are connected by the shortest side
            int64_t ikeep = ivert[(imin + 1) % 3];
            int64_t imerge = ivert[(imin + 2) % 3];
            
            // We always merge towards the more massive vertex
            if(mass[imerge] > mass[ikeep]) {
                int64_t itemp = ikeep;
                ikeep = imerge;
                imerge = itemp;
            }
            
            tomerge.push_back(pair<int64_t,int64_t>{ikeep, imerge});
            
            while(! tomerge.empty())
            {
                ikeep = tomerge.front().first;
                imerge = tomerge.front().second;
                tomerge.pop_front();
                
                myassert(ikeep != imerge, "This is weird... ikeep == imerge...\n");
                
                // Now merge the vertices
                TFloat mtot  = mass[ikeep] + mass[imerge];
                TFloat dx[2];
                wrapped_dx(&pos[2*ikeep], &pos[2*imerge], L, 2, dx);
                for(int ax=0; ax<2; ax++)
                    pos[2*ikeep + ax] += (mass[imerge]*dx[ax])/mtot;
                mass[ikeep] = mtot;
                mass[imerge] = 0.;

                idptr[imerge] = ikeep;

                // Update the triangle connectivity and remove degenerate triangles
                list<int64_t> defects;
                merge_trilists(ikeep, imerge, &part_to_tri[ikeep], &part_to_tri[imerge], tri, &defects);

                for (auto tri : part_to_tri[ikeep]) {
                    tris_to_check.push_back(tri);
                }
                
                // If we had defects, also merge those
                for (auto defect : defects) {
                    if(mass[ikeep] >= mass[defect])
                        tomerge.push_back(pair<int64_t,int64_t>{ikeep, defect});
                    else
                        tomerge.push_back(pair<int64_t,int64_t>{defect, ikeep});
                }
                
                // If we merged something we have to make sure to update it in the mergelist
                for (auto pair : tomerge) {
                    if(pair.first == imerge)
                        pair.first = ikeep;
                    if(pair.second == imerge)
                        pair.second = ikeep;
                }
            }
        }
    }

    return ;
}

template <typename TFloat>
void unfold2d_sorted(TFloat L, int npart, int ntri, TFloat* pos, TFloat* mass, int64_t* idptr, int64_t* tri)
{
    vector<list<int64_t>> part_to_tri;
    vector<bool> needs_check;
    
    part_to_tri.resize(npart);
    needs_check.resize(ntri);
    
    std::priority_queue<SimpStruct<TFloat>, vector<SimpStruct<TFloat> >, SimpCompare<TFloat>> queue;

    for (int64_t itri=0; itri<ntri; itri++)
    {
        needs_check[itri] = false;
        if(tri[3*itri] == -1) // deactivated triangle
            continue; 
        for(int ivert=0; ivert<3; ivert++) {
            part_to_tri[tri[3*itri+ivert]].push_back(itri);
        }
    }
    
    for (int64_t itri=0; itri<ntri; itri++)
    {
        int64_t *ivert = &tri[3*itri];
        if(ivert[0] == -1) // deactivated triangle
            continue; 
        
        TFloat area = triangle_parity(&pos[2*ivert[0]], &pos[2*ivert[1]], &pos[2*ivert[2]], L);

        if(area < 0.) {
            queue.push(SimpStruct<TFloat> {itri, area});
            needs_check[itri] = true;
        }
    }
    
    while (! queue.empty())
    {
        int64_t itri = queue.top().id;
        queue.pop();
        
        needs_check[itri] = false;
        
        int64_t *ivert = &tri[3*itri];
        
        if(ivert[0] == -1) // deactivated triangle
            continue; 
        
        TFloat parity = triangle_parity(&pos[2*ivert[0]], &pos[2*ivert[1]], &pos[2*ivert[2]], L);

        if(parity < 0.)
        {
            list< pair<int64_t, int64_t> > tomerge;
            
            
            // We keep the vertex opposite of the shortest side
            int64_t imin = triangle_shortest_side(&pos[2*ivert[0]], &pos[2*ivert[1]], &pos[2*ivert[2]], L);
            
            // We merge the two vertices that are connected by the shortest side
            int64_t ikeep = ivert[(imin + 1) % 3];
            int64_t imerge = ivert[(imin + 2) % 3];
            
            // We always merge towards the more massive vertex
            if(mass[imerge] > mass[ikeep]) {
                int64_t itemp = ikeep;
                ikeep = imerge;
                imerge = itemp;
            }
            
            tomerge.push_back(pair<int64_t,int64_t>{ikeep, imerge});
            
            while(! tomerge.empty())
            {
                ikeep = tomerge.front().first;
                imerge = tomerge.front().second;
                tomerge.pop_front();
                
                myassert(ikeep != imerge, "This is weird... ikeep == imerge...\n");
                
                // Now merge the vertices
                TFloat mtot  = mass[ikeep] + mass[imerge];
                TFloat dx[2];
                wrapped_dx(&pos[2*ikeep], &pos[2*imerge], L, 2, dx);
                for(int ax=0; ax<2; ax++)
                    pos[2*ikeep + ax] += (mass[imerge]*dx[ax])/mtot;
                mass[ikeep] = mtot;
                mass[imerge] = 0.;

                idptr[imerge] = ikeep;

                // Update the triangle connectivity and remove degenerate triangles
                list<int64_t> defects;
                merge_trilists(ikeep, imerge, &part_to_tri[ikeep], &part_to_tri[imerge], tri, &defects);

                for (int64_t itri : part_to_tri[ikeep]) {
                    if(not needs_check[itri]) // otherwise is already being checked anyways
                    {
                        int64_t *ivert = &tri[3*itri];
                        TFloat area = triangle_parity(&pos[2*ivert[0]], &pos[2*ivert[1]], &pos[2*ivert[2]], L);
                        if(area < 0.) {
                            queue.push(SimpStruct<TFloat> {itri, area});
                            needs_check[itri] = true;
                        }
                    }
                }
                
                // If we had defects, also merge those
                for (auto defect : defects) {
                    if(mass[ikeep] >= mass[defect])
                        tomerge.push_back(pair<int64_t,int64_t>{ikeep, defect});
                    else
                        tomerge.push_back(pair<int64_t,int64_t>{defect, ikeep});
                }
                
                // If we merged something we have to make sure to update it in the mergelist
                for (auto pair : tomerge) {
                    if(pair.first == imerge)
                        pair.first = ikeep;
                    if(pair.second == imerge)
                        pair.second = ikeep;
                }
            }
        }
    }

    return ;
}

template <typename TFloat>
void unfold3d(TFloat L, int npart, int ntets, TFloat* pos, TFloat* mass, int64_t* idptr, int64_t* tets)
{
    vector<list<int64_t>> part_to_tets;
    
    vector<bool> needs_check;
    
    part_to_tets.resize(npart);
    needs_check.resize(ntets);

    for (int64_t itet=0; itet<ntets; itet++)
    {
        needs_check[itet] = false;
        if(tets[4*itet] == -1) // deactivated tet
            continue; 
        for(int ivert=0; ivert<4; ivert++) {
            part_to_tets[tets[4*itet+ivert]].push_back(itet);
        }
    }
    
    list<int64_t> tets_to_check;
    for (int64_t itet=0; itet<ntets; itet++)
    {
        int64_t *ivert = &tets[4*itet];
        if(ivert[0] == -1) // deactivated triangle
            continue; 
        
        TFloat parity = tet_parity(&pos[3*ivert[0]], &pos[3*ivert[1]], &pos[3*ivert[2]], &pos[3*ivert[3]], L);
        if(parity < 0.) {
            tets_to_check.push_back(itet);
            needs_check[itet] = true;
        }
    }
    printf("ntocheck: %lld\n", tets_to_check.size());
    fflush(stdout);
    
    int64_t nchecked=0, nfound=0;
    
    while (! tets_to_check.empty())
    {
        nchecked++;
        
        int itet = tets_to_check.front();
        tets_to_check.pop_front();
        
        needs_check[itet] = false;
        
        int64_t *ivert = &tets[4*itet];
        if(ivert[0] == -1) // deactivated triangle
            continue;
        
        TFloat parity = tet_parity(&pos[3*ivert[0]], &pos[3*ivert[1]], &pos[3*ivert[2]], &pos[3*ivert[3]], L);

        if(parity < 0.)
        {
            nfound++;
            
            list< pair<int64_t, int64_t> > tomerge;
            
            // We merge the two vertices that are connected by the shortest side
            pair<int64_t, int64_t> edge = tet_shortest_side(&pos[3*ivert[0]], &pos[3*ivert[1]], &pos[3*ivert[2]], &pos[3*ivert[3]], L);
            int64_t ikeep=ivert[edge.first], imerge=ivert[edge.second];
            
            // We always merge towards the more massive vertex
            if(mass[imerge] > mass[ikeep]) {
                int64_t itemp = ikeep;
                ikeep = imerge;
                imerge = itemp;
            }
            
            tomerge.push_back(pair<int64_t,int64_t>{ikeep, imerge});
            
            while(! tomerge.empty())
            {
                ikeep = tomerge.front().first;
                imerge = tomerge.front().second;
                tomerge.pop_front();
                
                myassert(ikeep != imerge, "This is weird... ikeep == imerge...\n");
                
                myassert(mass[ikeep] > 0., "This is weird... singular mass...\n");
                
                // Now merge the vertices
                TFloat mtot  = mass[ikeep] + mass[imerge];
                TFloat dx[3];
                wrapped_dx(&pos[3*ikeep], &pos[3*imerge], L, 3, dx);
                for(int ax=0; ax<3; ax++)
                    pos[3*ikeep + ax] += (mass[imerge]*dx[ax])/mtot;
                
                mass[ikeep] = mtot;
                mass[imerge] = 0.;
                idptr[imerge] = ikeep;
                
                // Update the triangle connectivity and remove degenerate triangles
                //printf("n1 %lld n2 %lld ", part_to_tets[ikeep].size(), part_to_tets[imerge].size());
                
                list<pair<int64_t,int64_t> > defects;
                merge_tetlists(ikeep, imerge, &part_to_tets[ikeep], &part_to_tets[imerge], tets, &defects);
                
                //printf("n3 %lld ncheck %lld", part_to_tets[ikeep].size(), tets_to_check.size());
                //printf("nchecknew: %lld\n", part_to_tets[ikeep].size());
                

                for (auto tet : part_to_tets[ikeep]) {
                    if(not needs_check[tet]){
                        tets_to_check.push_back(tet);
                        needs_check[tet] = true;
                    }
                }
                
                //printf("ncheck+ %lld\n", tets_to_check.size());
                //fflush(stdout);
                /*
                // If we had defects, also merge those
                for (auto defect : defects) {
                    if(mass[ikeep] >= mass[defect])
                        tomerge.push_back(pair<int64_t,int64_t>{ikeep, defect});
                    else
                        tomerge.push_back(pair<int64_t,int64_t>{defect, ikeep});
                }
                
                // If we merged something we have to make sure to update it in the mergelist
                for (auto pair : tomerge) {
                    if(pair.first == imerge)
                        pair.first = ikeep;
                    if(pair.second == imerge)
                        pair.second = ikeep;
                }*/
            }
        }
    }
    
    printf("nchecked: %lld, nfound %lld, ntot %lld\n", nchecked, nfound, ntets);
    fflush(stdout);
}



void merge_tetlists_new(int64_t ikeep, int64_t imerge, vector<int64_t> &tetlist_keep, vector<int64_t> &tetlist_merge, int64_t *tets, list< pair<int64_t,int64_t> > *defects, Timer* timer)
{
    list<pair<int64_t, int64_t> > degenerate_tets;
    
    for (int64_t i=0; i<tetlist_merge.size(); i++) {
        int64_t itet = tetlist_merge[i];
        int64_t *ivert = &tets[4*itet]; 
        
        /* Remove degenerate tets */
        if(find_in(ivert, 4, ikeep)  >= 0) {
            ivert[0] = ivert[1] = ivert[2] = ivert[3] = -1;
            eraseElement(tetlist_merge, i);
            i -= 1; // have to jump back, since we swapped the last element to location i
        }
        else {
            // replace imerge by ikeep in non-degenerate tets 
            for(int i=0; i<4; i++)
            {
                if(ivert[i] == imerge)
                    ivert[i] = ikeep;
            }
        }
    }
    
    /* Merge the lists */
    tetlist_keep.insert(tetlist_keep.end(), tetlist_merge.begin(), tetlist_merge.end());
    tetlist_merge.clear();
    
    if(timer != nullptr)
        timer->measureDT("LoopMergeList");
}


template <typename TFloat>
void unfold3d_sorted(TFloat L, int npart, int ntets, TFloat* pos, TFloat* mass, int64_t* idptr, int64_t* tets, int mode)
{
    Timer mytimer(true);
    
    vector<vector<int64_t>> part_to_tets;
    vector<bool> needs_check;
    
    part_to_tets.resize(npart);
    needs_check.resize(ntets);

    std::priority_queue<SimpStruct<TFloat>, vector<SimpStruct<TFloat> >, SimpCompare<TFloat>> queue; 
    
    for (int64_t itet=0; itet<ntets; itet++)
    {
        needs_check[itet] = false;
        if(tets[4*itet] == -1) // deactivated tet
            continue; 
        for(int ivert=0; ivert<4; ivert++) {
            part_to_tets[tets[4*itet+ivert]].push_back(itet);
        }
    }
    
    for (int64_t itet=0; itet<ntets; itet++)
    {
        int64_t *ivert = &tets[4*itet];
        if(ivert[0] == -1) // deactivated tet
            continue; 
        
        TFloat area = tet_parity(&pos[3*ivert[0]], &pos[3*ivert[1]], &pos[3*ivert[2]], &pos[3*ivert[3]], L);
        if(area < 0.) {
            queue.push(SimpStruct<TFloat> {itet, area});
            needs_check[itet] = true;
        }
    }
    printf("ntocheck: %lld\n", queue.size());
    fflush(stdout);
    
    int64_t nchecked=0, nfound=0, ndefect=0;
    
    mytimer.measureDT("Preparations");
    
    while (! queue.empty())
    {
        
        nchecked++;
        
        int64_t itet = queue.top().id;
        queue.pop();
        
        needs_check[itet] = false;
        int64_t *ivert = &tets[4*itet];
        
        if(ivert[0] < 0) // deactivated tet
            continue;

        TFloat parity = tet_parity(&pos[3*ivert[0]], &pos[3*ivert[1]], &pos[3*ivert[2]], &pos[3*ivert[3]], L);
        
        mytimer.measureDT("Parity Check");

        if(parity < 0.)
        {
            nfound++;
            
            list< pair<int64_t, int64_t> > tomerge;
            
            // We merge the two vertices that are connected by the shortest side
            pair<int64_t, int64_t> edge = tet_shortest_side(&pos[3*ivert[0]], &pos[3*ivert[1]], &pos[3*ivert[2]], &pos[3*ivert[3]], L);
            tomerge.push_back(pair<int64_t,int64_t>{ivert[edge.first], ivert[edge.second]});
            
            while(! tomerge.empty())
            {
                int64_t ikeep = tomerge.front().first;
                int64_t imerge = tomerge.front().second;
                tomerge.pop_front();
                
                // We always merge towards the more massive vertex
                if(mass[imerge] > mass[ikeep]) {
                    int64_t itemp = ikeep;
                    ikeep = imerge;
                    imerge = itemp;
                }
                
                myassert(ikeep != imerge, "This is weird... ikeep == imerge...\n");
                
                myassert(mass[ikeep] > 0., "This is weird... singular mass...\n");
                
                // Now merge the vertices
                TFloat mtot  = mass[ikeep] + mass[imerge];
                TFloat dx[3];
                wrapped_dx(&pos[3*ikeep], &pos[3*imerge], L, 3, dx);
                for(int ax=0; ax<3; ax++)
                    pos[3*ikeep + ax] += (mass[imerge]*dx[ax])/mtot;
                
                mass[ikeep] = mtot;
                mass[imerge] = 0.;
                idptr[imerge] = ikeep;
                
                mytimer.measureDT("Merge");
                
                // Update the triangle connectivity and remove degenerate triangles
                list<pair<int64_t,int64_t> > defects;
                merge_tetlists_new(ikeep, imerge, part_to_tets[ikeep], part_to_tets[imerge], tets, &defects, &mytimer);

                mytimer.measureDT("MergeLists");

                for (int64_t i=0; i<part_to_tets[ikeep].size(); i++) {
                    int64_t tet = part_to_tets[ikeep][i];

                    int64_t *ivert = &tets[4*tet];
                    
                    if(ivert[0] < 0) // deactivated tet, which was not removed from the list yet
                    {
                        eraseElement(part_to_tets[ikeep], i);
                        i -= 1;
                        continue;
                    }
                    
                    if(not needs_check[tet]) // Only add the simplex if it is not already in the check-list
                    {
                        TFloat area = tet_parity(&pos[3*ivert[0]], &pos[3*ivert[1]], &pos[3*ivert[2]], &pos[3*ivert[3]], L);
                        
                        if(area < 0.) {
                            queue.push(SimpStruct<TFloat> {tet, area});
                            needs_check[tet] = true;
                        }
                    }
                }
                
                mytimer.measureDT("AppendLists");
                
                if((mode == 5) && (defects.size() > 0)) {
                    ndefect += defects.size();
                    printf("Have %d defects\n", defects.size());
                    
                    for (auto defect : defects) {
                        tomerge.push_back(pair<int64_t,int64_t>{ikeep, defect.first});
                        tomerge.push_back(pair<int64_t,int64_t>{ikeep, defect.second});
                    }
                }
                
                if(mode >= 5) {
                    // If we merged something we have to make sure to update it in the mergelist
                    for (auto pair : tomerge) {
                        if(pair.first == imerge)
                            pair.first = ikeep;
                        if(pair.second == imerge)
                            pair.second = ikeep;
                    }
                }
                
                mytimer.measureDT("AppendDefects");
            }
        }
    }

    printf("nchecked: %lld, nfound %lld, ntot %lld, ndefect %lld\n", nchecked, nfound, ntets, ndefect);
    mytimer.printResults();
    fflush(stdout);
}


template void unfold2d(float L, int npart, int ntri, float* pos, float* mass, int64_t* idptr, int64_t* tri);
template void unfold2d_sorted(float L, int npart, int ntri, float* pos, float* mass, int64_t* idptr, int64_t* tri);
template void unfold3d(float L, int npart, int ntets, float* pos, float* mass, int64_t* idptr, int64_t* tets);
template void unfold3d_sorted(float L, int npart, int ntets, float* pos, float* mass, int64_t* idptr, int64_t* tets, int mode);