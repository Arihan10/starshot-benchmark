const debug = process.env.NEXT_PUBLIC_ENABLE_DEBUG;

/**
 * The font/colour labs and every /debug route. On in development; a production
 * deploy has to ask for them with NEXT_PUBLIC_ENABLE_DEBUG=1.
 */
export const DEBUG_ENABLED =
	debug === "1" || (debug !== "0" && process.env.NODE_ENV === "development");

/** #TODO temporary — interior pano tours are off until the captures are fixed. */
export const TOURS_ENABLED: boolean = false;
