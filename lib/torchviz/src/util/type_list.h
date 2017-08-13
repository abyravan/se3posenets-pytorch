#ifndef L2S_TYPE_LIST_H_
#define L2S_TYPE_LIST_H_

namespace l2s {

struct NullType { };

template <class H, class T>
struct TypeList {
    typedef H head;
    typedef T tail;
};

// construct a typelist
#define TYPELIST1(T1) l2s::TypeList<T1, l2s::NullType>
#define TYPELIST2(T1, T2) l2s::TypeList<T1, TYPELIST1(T2) >
#define TYPELIST3(T1, T2, T3) l2s::TypeList<T1, TYPELIST2(T2, T3) >

// get the the length of a typelist
template <typename TL> struct Length;
template <> struct Length<NullType> {
    enum { value = 0 };
};
template <typename T, typename U>
struct Length< TypeList<T,U> > {
    enum { value = 1 + Length<U>::value };
};

// indexed access into a typelist
template <typename TL, unsigned int index> struct TypeAt;

template <typename Head, typename Tail>
struct TypeAt<TypeList<Head,Tail>,0> {
    typedef Head result;
};

template <typename Head, typename Tail, unsigned int index>
struct TypeAt<TypeList<Head,Tail>,index> {
    typedef typename TypeAt<Tail,index-1>::result result;
};

// forgiving indexed access into a typelist
template <typename TL, unsigned int index, typename Default>
struct TypeAtNonStrict {
    typedef Default result;
};

template <typename Head, typename Tail, typename Default>
struct TypeAtNonStrict<TypeList<Head,Tail>,0,Default> {
    typedef Head result;
};

template <typename Head, typename Tail, unsigned int index, typename Default>
struct TypeAtNonStrict<TypeList<Head,Tail>,index,Default> {
    typedef typename TypeAtNonStrict<Tail,index-1,Default>::result result;
};

// erase a type from a typelist
template <typename TL, typename ToErase> struct Erase;

template <typename ToErase>
struct Erase<NullType,ToErase> {
    typedef NullType result;
};

template <typename ToErase, typename Tail>
struct Erase<TypeList<ToErase,Tail>,ToErase> {
    typedef Tail result;
};

template <typename Head, typename Tail, typename ToErase>
struct Erase<TypeList<Head,Tail>,ToErase> {
    typedef TypeList<Head,typename Erase<Tail,ToErase>::result> result;
};

// erase all appearances of a type from a typelist
template <typename TL, typename ToErase> struct EraseAll;

template <typename ToErase>
struct EraseAll<NullType,ToErase> {
    typedef NullType result;
};

template <typename Tail, typename ToErase>
struct EraseAll<TypeList<ToErase,Tail>,ToErase> {
    typedef typename EraseAll<Tail,ToErase>::result result;
};

template <typename Head, typename Tail, typename ToErase>
struct EraseAll<TypeList<Head,Tail>,ToErase> {
    typedef TypeList<Head,typename EraseAll<Tail,ToErase>::result > result;
};

// remove duplicates from a typelist
template <typename TL> struct NoDuplicates;

template <> struct NoDuplicates<NullType> {
    typedef NullType result;
};

template <typename Head, typename Tail>
struct NoDuplicates< TypeList<Head,Tail> > {
private:
    typedef typename NoDuplicates<Tail>::result L1;
    typedef typename Erase<L1,Head>::result L2;
public:
    typedef TypeList<Head,L2> result;
};

// generate a scattered inheritance hierarchy for frame buffers
template <typename TL, template <typename> class Unit>
class GenScatteredFBHierarchy;

template <typename T1, typename T2, template <typename> class Unit>
class GenScatteredFBHierarchy<TypeList<T1,T2>,Unit>
        : public GenScatteredFBHierarchy<T1, Unit>,
          public GenScatteredFBHierarchy<T2, Unit> {
public:
    typedef TypeList<T1, T2> TList;
    typedef GenScatteredFBHierarchy<T1, Unit> LeftBase;
    typedef GenScatteredFBHierarchy<T2, Unit> RightBase;

    explicit GenScatteredFBHierarchy(const int width, const int height)
        : LeftBase(width,height), RightBase(width,height) { }
};

template <typename AtomicType, template <typename> class Unit>
class GenScatteredFBHierarchy : public Unit<AtomicType> {
private:
    typedef Unit<AtomicType> LeftBase;
public:
    explicit GenScatteredFBHierarchy(const int width, const int height)
        : LeftBase(width, height) { }
};

template <template <typename> class Unit>
class GenScatteredFBHierarchy<NullType, Unit> {
public:

    explicit GenScatteredFBHierarchy(const int width, const int height) { }
};

// generate a scattered generic inheritance hierarchy
template <typename TL, template <typename> class Unit>
class GenScatteredHierarchy;

template <typename T1, typename T2, template <typename> class Unit>
class GenScatteredHierarchy<TypeList<T1,T2>,Unit>
        : public GenScatteredHierarchy<T1, Unit>,
          public GenScatteredHierarchy<T2, Unit> {
public:
    typedef TypeList<T1, T2> TList;
    typedef GenScatteredHierarchy<T1, Unit> LeftBase;
    typedef GenScatteredHierarchy<T2, Unit> RightBase;
};

template <typename AtomicType, template <typename> class Unit>
class GenScatteredHierarchy : public Unit<AtomicType> {
private:
    typedef Unit<AtomicType> LeftBase;
};

template <template <typename> class Unit>
class GenScatteredHierarchy<NullType, Unit> { };

} // namespace l2s

#endif // L2S_TYPE_LIST_H_
