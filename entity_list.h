#ifndef __entityList_h__
#define __entityList_h__

#include "entity.h"

class entity_list : public entity {

    private:
        



    public:
        __device__ entity_list();
        __device__ entity_list(entity**, int);
       __device__ virtual bool hit(const ray&, float, float, hit_record&) const;

        entity** list;
        int length;

};


#endif