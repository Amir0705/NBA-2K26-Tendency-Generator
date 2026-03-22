declare module "lucide-react";

declare module "next/navigation" {
  export function useRouter(): any;
  export function usePathname(): string;
  export function useSearchParams(): URLSearchParams;
  export function redirect(path: string): never;
}

declare module "next/server" {
  export const NextResponse: any;
  export type NextRequest = any;
}
