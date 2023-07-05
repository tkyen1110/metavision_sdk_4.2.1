
#ifndef METAVISION_HAL_EXPORT_H
#define METAVISION_HAL_EXPORT_H

#ifdef METAVISION_HAL_STATIC_DEFINE
#  define METAVISION_HAL_EXPORT
#  define METAVISION_HAL_NO_EXPORT
#else
#  ifndef METAVISION_HAL_EXPORT
#    ifdef metavision_hal_EXPORTS
        /* We are building this library */
#      define METAVISION_HAL_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define METAVISION_HAL_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef METAVISION_HAL_NO_EXPORT
#    define METAVISION_HAL_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef METAVISION_HAL_DEPRECATED
#  define METAVISION_HAL_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef METAVISION_HAL_DEPRECATED_EXPORT
#  define METAVISION_HAL_DEPRECATED_EXPORT METAVISION_HAL_EXPORT METAVISION_HAL_DEPRECATED
#endif

#ifndef METAVISION_HAL_DEPRECATED_NO_EXPORT
#  define METAVISION_HAL_DEPRECATED_NO_EXPORT METAVISION_HAL_NO_EXPORT METAVISION_HAL_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef METAVISION_HAL_NO_DEPRECATED
#    define METAVISION_HAL_NO_DEPRECATED
#  endif
#endif

        #ifndef METAVISION_HAL_EXTERN_EXPORT 
        #  define METAVISION_HAL_EXTERN_EXPORT METAVISION_HAL_EXPORT
        #endif
#endif /* METAVISION_HAL_EXPORT_H */
