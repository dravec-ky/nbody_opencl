//Largely inspired by https://github.com/bneukom/gpu-nbody/blob/master/kernels/nbody/buildtree.cl
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#ifdef HAS__debug
    #define DEBUG(id,x,val) _debug[id*8+x] = val
#else 
    #define DEBUG(id,x,val) do {} while (0)
#endif

kernel void build_octree( 
#ifdef SINGLE_WORKGROUP
                 int num_worksets, 
                 global const int *worksets_begins, 
                 global const int *worksets_lengths, 
#else
                 int worksets_begin, 
                 int worksets_length, 
#endif
                 int _pos_length, 
                 global float * _pos ,
                 int _cell_pos_length, 
                 global float * _cell_pos ,
                 int _mass_length, 
                 global float * _mass ,
                 int _cell_mass_length, 
                 global float * _cell_mass ,
                 int _cell_tree_length, 
                 global int * _cell_tree ,
                 int _cell_body_count_length, 
                 global int * _cell_body_count ,
#ifdef HAS__cell_start
                 int _cell_start_length, 
                 global int * _cell_start ,
#endif
#ifdef HAS__cell_sorted
                 int _cell_sorted_length, 
                 global int * _cell_sorted ,
#endif
                 int _g_acc_length, 
                 global float * _g_acc ,
                 int _bottom_length, 
                 global int * _bottom ,
                 int _radius_length, 
                 global float * _radius ,
#ifdef HAS__debug
                 int _debug_length, 
                 global float * _debug ,
#endif
#ifdef HAS__octree_depth
                 int _octree_depth_length, 
                 global int * _octree_depth ,
#endif
#ifdef HAS__octree_pos
                 int _octree_pos_length, 
                 global float * _octree_pos ,
#endif
#ifdef HAS__preview_id
                 int _preview_id_length, 
                 global int * _preview_id ,
#endif
#ifdef HAS__preview_switch
                 int _preview_switch_length, 
                 global int * _preview_switch ,
#endif
                 int _max_depth_length, 
                 global int * _max_depth ,
                 int _incr_length, 
                 global int * _incr ,
                 int _theta_length, 
                 global float * _theta ,
                 int _softening_length, 
                 global float * _softening ,
                 int _world_scale_length, 
                 global float * _world_scale,
                 int _mass_scale_length, 
                 global float * _mass_scale,
                 int _g_temp_length, 
                 global float * _g_temp 
)
{
    
#ifdef SINGLE_WORKGROUP
    for(int i=0; i < num_worksets; i++)
    {
        int worksets_begin = worksets_begins[i];
        int worksets_length = worksets_lengths[i];
//            barrier(CLK_GLOBAL_MEM_FENCE);
#else
    {  
#endif
//Index initialzation
    int idx = get_global_id(0);
    const int local_idx = idx;
    if (idx >= worksets_length)
        return;
    idx += worksets_begin;
    
    int body_id = idx;
#ifdef SINGLE_WORKGROUP
    DEBUG(body_id,0,-511);
#else
    DEBUG(body_id,0,-510);
#endif

    int depth;
    int local_max_depth = 0;
    const int NUM_BODIES = _cell_pos_length-1;
    //maximum number of cells + bodies
    const int NUM_NODES = NUM_BODIES*2+1;
    const int NUM_CELLS = 8;
    const int MAX_LOOP = 10000;

    const int CELL_OFFSET = NUM_BODIES+1;
    const int MAX_DEPTH = *_max_depth;

    float3 root = vload3(_cell_pos_length-1, _cell_pos);
    int child_path;
    int parent;
    float3 body;
    float mass;
    float3 child_pos;
    bool finished = false;
    
    //-1 - unoccupied, -2 - locked, >=0 - parent index
    int child_idx;
    //If operating in worksets, start point loop here:
    //Loading new body, starting at the root
    int j=0;

    while(finished==false && j<MAX_LOOP)
    {
        float current_radius;
        body = vload3(idx, _pos);

        //parent is root
        parent = NUM_NODES;
        depth = 1;
        current_radius = *_radius;
        child_path = 0;
        
        //Set path to follow
        if(root.x < _pos[idx*3]) child_path = 1;
        if(root.y < _pos[idx*3+1]) child_path += 2;
        if(root.z < _pos[idx*3+2]) child_path += 4;

        child_idx = _cell_tree[NUM_CELLS * (parent-CELL_OFFSET) + child_path];
        
        bool is_cell = child_idx >= NUM_BODIES;

        //Goes down the tree until it finds locked/unoccupied/occupied parent 
        while(is_cell && depth<MAX_DEPTH)
        {
            ++depth;
            parent = child_idx;
            current_radius *= 0.5;
            
            child_path = 0;
            
            if(_cell_pos[(parent-CELL_OFFSET)*3] < body.x) child_path = 1;
            if(_cell_pos[(parent-CELL_OFFSET)*3+1] < body.y) child_path += 2;
            if(_cell_pos[(parent-CELL_OFFSET)*3+2] < body.z) child_path += 4;

            child_idx = _cell_tree[NUM_CELLS * (parent-CELL_OFFSET) + child_path];
            is_cell = child_idx > NUM_BODIES;
        }

        if(child_idx >= -1)
        {
            int locked = NUM_CELLS * (parent-CELL_OFFSET) + child_path;
            //Locking mechanism
            //atom_cmpxchg(v1,v2,v3) - Compares v1 (old) and v2, if they're equal,
            //switches value1 to v3 - allows only one work item to use the parent at a time
            
            if( child_idx == atom_cmpxchg(&_cell_tree[locked], child_idx, -idx-2))
            {
                //If unoccupied, just insert itself
                if(child_idx == NULL_BODY)
                {
                    _cell_tree[locked] = idx;
                }
                //If occupied, the cell has to subdivide further
                else
                {
                    int patch = -1;
                    if(is_cell)
                    {
                        const int cell_idx = child_idx-CELL_OFFSET;
                        patch = cell_idx;
                        const float new_mass = _cell_mass[cell_idx] + _mass[idx];
                        const float mass_mult = (new_mass==0.0)?0.5:1/new_mass;
                        _cell_pos[(cell_idx)*3] = (_cell_pos[cell_idx*3]*_cell_mass[cell_idx] + body.x*_mass[idx])*mass_mult;
                        _cell_pos[(cell_idx)*3+1] = (_cell_pos[cell_idx*3+1]*_cell_mass[cell_idx] + body.y*_mass[idx])*mass_mult;
                        _cell_pos[(cell_idx)*3+2] = (_cell_pos[cell_idx*3+2]*_cell_mass[cell_idx] + body.z*_mass[idx])*mass_mult;
                        _cell_mass[cell_idx] = new_mass;
                        atom_inc(&_cell_body_count[cell_idx]);
                    }
                    else
                    {
                        int k=0;
                        do
                        {
                            //decreases the floor of octree - when running parents
                            //in parallel it prevents them from overwriting each other
                            //range of <cell, cell + 7> is free for us to work on
                            //function returns pre-subtracted _bottom, need to subtract again locally
                            depth++;
                            int cell_idx = atom_dec(_bottom)-1;

                            int cell = cell_idx + CELL_OFFSET;

                            //Hit bottom, if happens decrease maximum depth
                            if(cell_idx < 0)
                            {
                                *_bottom = NUM_NODES;
                                DEBUG(body_id,0,-666);
                                return;
                                child_path = 0; //escape loop
                                child_idx = NULL_BODY;
                            }

                            
                            //patch will always equal 1st while loop cell
                            patch = max(patch, cell);
                            //Calculate cell offsets using bitwise operators
                            //e.g & 1 - every 2nd cell
                            if(depth>=MAX_DEPTH)
                            {
                                #ifdef HAS__octree_depth                               
                                    float x = (child_path & 1) * current_radius;
                                    float y = ((child_path >> 1) & 1) * current_radius;
                                    float z = ((child_path >> 2) & 1) * current_radius;

                                    current_radius *= 0.5f;

                                    x = _octree_pos[(cell_idx)*3] = _octree_pos[(parent-CELL_OFFSET)*3] - current_radius + x;
                                    y = _octree_pos[(cell_idx)*3+1] = _octree_pos[(parent-CELL_OFFSET)*3+1] - current_radius + y;
                                    z = _octree_pos[(cell_idx)*3+2] = _octree_pos[(parent-CELL_OFFSET)*3+2] - current_radius + z;

                                    _octree_depth[(cell_idx)] = depth;
                                #endif

                                #pragma unroll 8
						            for (int l = 0; l < NUM_CELLS; l++) _cell_tree[NUM_CELLS * cell_idx + l] = -1;
                                _cell_mass[cell_idx] = _mass[child_idx]+_mass[idx];
                                const float mass_mult = (_cell_mass[cell_idx]==0.0)?0.5:1/_cell_mass[cell_idx];
                                _cell_pos[(cell_idx)*3] =  (_pos[child_idx*3]*_mass[child_idx] + body.x*_mass[idx])*mass_mult;
                                _cell_pos[(cell_idx)*3+1] =  (_pos[child_idx*3+1]*_mass[child_idx] + body.y*_mass[idx])*mass_mult;
                                _cell_pos[(cell_idx)*3+2] =  (_pos[child_idx*3+2]*_mass[child_idx] + body.z*_mass[idx])*mass_mult;
                                _cell_body_count[cell_idx] = 2;
                                
                                child_idx = NULL_BODY; //escape loop
                                parent = cell;
                                idx = cell;
                            }
                            else
                            {
                                float x = (child_path & 1) * current_radius;
                                float y = ((child_path >> 1) & 1) * current_radius;
                                float z = ((child_path >> 2) & 1) * current_radius;

                                current_radius *= 0.5f;

                                //Not needed because its set in wrangle?
                                //Setting it here allows us to not have to write cell_mass and body count to the GPU every frame :)
                                _cell_mass[cell_idx] = -1.0f;
                                _cell_body_count[cell_idx] = 0;

                                //Calculate final cell positions - subtract halved radius
                                //to get to the child cell center and add the offset
                                x = _cell_pos[(cell_idx)*3] = _cell_pos[(parent-CELL_OFFSET)*3] - current_radius + x;
                                y = _cell_pos[(cell_idx)*3+1] = _cell_pos[(parent-CELL_OFFSET)*3+1] - current_radius + y;
                                z = _cell_pos[(cell_idx)*3+2] = _cell_pos[(parent-CELL_OFFSET)*3+2] - current_radius + z;

                                #ifdef HAS__octree_depth
                                    _octree_depth[(cell_idx)] = depth;
                                    _octree_pos[(cell_idx)*3] = _cell_pos[(cell_idx)*3];
                                    _octree_pos[(cell_idx)*3+1] = _cell_pos[(cell_idx)*3+1];
                                    _octree_pos[(cell_idx)*3+2] = _cell_pos[(cell_idx)*3+2];
                                #endif

                                #pragma unroll 8
						            for (int l = 0; l < NUM_CELLS; l++) _cell_tree[NUM_CELLS * cell_idx + l] = -1;

                                if(patch!=cell)
                                {
                                    _cell_tree[NUM_CELLS * (parent-CELL_OFFSET) + child_path] = cell;
                                }

                                //Assign new path to body that occupied the parent cell
                                child_path = 0;
                                if(x < _pos[child_idx*3]) child_path = 1;
                                if(y < _pos[child_idx*3+1]) child_path += 2;
                                if(z < _pos[child_idx*3+2]) child_path += 4;

                                _cell_tree[NUM_CELLS* cell_idx + child_path] = child_idx;

                                //Check for new index for our current body
                                parent = cell;
                                child_path = 0;
                                if(x < body.x) child_path = 1;
                                if(y < body.y) child_path += 2;
                                if(z < body.z) child_path += 4;
                                child_idx = _cell_tree[NUM_CELLS* (parent-CELL_OFFSET) + child_path];
                                
                            }
                            ++k;
                        }
                        while(child_idx >= 0 && k<MAX_LOOP);
                        
                        //Assign new index once the body runs into unoccupied space
                        _cell_tree[NUM_CELLS* (parent-CELL_OFFSET) + child_path] = idx;
                    }
                    //Sync workers
                    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);

                    //connect new tree to the locked parent
                    _cell_tree[locked] = patch;
                }
                local_max_depth = max(depth, local_max_depth);
                finished = true;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        ++j;
    }
#ifdef SINGLE_WORKGROUP
    }
#else
    }
#endif
//END MAIN BODY
//
}